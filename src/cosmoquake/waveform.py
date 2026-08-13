"""Reading seismic traces from the formats used by Apollo and InSight.

The two missions ship the same physical quantity under different column
names, so a single reader normalises both into a :class:`Waveform`.

======  ===========================  ==================  =================
Body    File                         Time column         Velocity column
======  ===========================  ==================  =================
Moon    ``xa.s12.00.mhz.*.csv``      ``time_rel(sec)``   ``velocity(m/s)``
Mars    ``XB.ELYSE.02.BHV.*.csv``    ``rel_time(sec)``   ``velocity(c/s)``
======  ===========================  ==================  =================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Column names that hold relative time in seconds, in order of preference.
TIME_COLUMNS = ("time_rel(sec)", "rel_time(sec)", "time(sec)")

#: Column names that hold ground velocity, in order of preference.
VELOCITY_COLUMNS = ("velocity(m/s)", "velocity(c/s)", "velocity")

#: Extensions understood by :func:`read_waveform`.
CSV_SUFFIXES = (".csv",)
MSEED_SUFFIXES = (".mseed", ".miniseed", ".msd")


class WaveformError(ValueError):
    """Raised when a file cannot be interpreted as a seismic trace."""


@dataclass(frozen=True)
class Waveform:
    """A uniformly sampled velocity trace.

    Attributes:
        time: Relative time of each sample, in seconds since the start of
            the record.
        velocity: Ground velocity at each sample.
        source: Path the trace was read from, for provenance in reports.
    """

    time: np.ndarray
    velocity: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        if self.time.shape != self.velocity.shape:
            raise WaveformError(
                f"time and velocity lengths differ: "
                f"{self.time.shape[0]} vs {self.velocity.shape[0]}"
            )
        if self.time.size < 2:
            raise WaveformError("a waveform needs at least two samples")

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def sample_spacing(self) -> float:
        """Seconds between consecutive samples.

        Uses the median difference rather than ``time[1] - time[0]`` so a
        single duplicated or dropped timestamp cannot distort the frequency
        axis of every downstream spectrum.
        """
        spacing = float(np.median(np.diff(self.time)))
        if spacing <= 0:
            raise WaveformError("time column is not strictly increasing")
        return spacing

    @property
    def sampling_rate(self) -> float:
        """Samples per second."""
        return 1.0 / self.sample_spacing

    @property
    def duration(self) -> float:
        """Length of the record in seconds."""
        return float(self.time[-1] - self.time[0])

    def after(self, start_time: float) -> Waveform:
        """Return the portion of the trace at or after ``start_time`` seconds.

        Raises:
            WaveformError: If fewer than two samples remain, which would
                leave nothing to transform.
        """
        mask = self.time >= start_time
        if int(np.count_nonzero(mask)) < 2:
            raise WaveformError(
                f"no usable samples after t={start_time:.2f}s "
                f"(record ends at t={self.time[-1]:.2f}s)"
            )
        return Waveform(self.time[mask], self.velocity[mask], self.source)


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
    """Return the first candidate column present in ``frame``."""
    for name in candidates:
        if name in frame.columns:
            return name
    raise WaveformError(
        f"no {kind} column found; expected one of {list(candidates)}, "
        f"got {list(frame.columns)}"
    )


def read_csv_waveform(path: str | Path) -> Waveform:
    """Read an Apollo- or InSight-style seismic CSV export."""
    path = Path(path)
    frame = pd.read_csv(path)
    time_col = _pick_column(frame, TIME_COLUMNS, "time")
    velocity_col = _pick_column(frame, VELOCITY_COLUMNS, "velocity")

    frame = frame[[time_col, velocity_col]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna()
    return Waveform(
        time=frame[time_col].to_numpy(dtype=float),
        velocity=frame[velocity_col].to_numpy(dtype=float),
        source=str(path),
    )


def read_mseed_waveform(path: str | Path) -> Waveform:
    """Read the first trace of a miniSEED volume."""
    from obspy import read  # imported lazily; obspy is slow to import

    path = Path(path)
    stream = read(str(path))
    if len(stream) == 0:
        raise WaveformError(f"{path} contains no traces")

    trace = stream.traces[0]
    velocity = np.asarray(trace.data, dtype=float)
    spacing = float(trace.stats.delta)
    time = np.arange(velocity.size, dtype=float) * spacing
    return Waveform(time=time, velocity=velocity, source=str(path))


def read_waveform(path: str | Path) -> Waveform:
    """Read a seismic trace, dispatching on file extension.

    Args:
        path: A ``.csv`` or ``.mseed`` seismic record.

    Raises:
        WaveformError: If the extension is unknown or the contents cannot be
            parsed as a trace.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        return read_csv_waveform(path)
    if suffix in MSEED_SUFFIXES:
        return read_mseed_waveform(path)
    raise WaveformError(
        f"unsupported file type {suffix!r}; "
        f"expected one of {list(CSV_SUFFIXES + MSEED_SUFFIXES)}"
    )
