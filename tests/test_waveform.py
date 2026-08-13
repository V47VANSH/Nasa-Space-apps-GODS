"""Reading and slicing seismic traces."""

from __future__ import annotations

import numpy as np
import pytest

from cosmoquake.waveform import (
    Waveform,
    WaveformError,
    read_csv_waveform,
    read_waveform,
)


def test_reads_insight_column_names(mars_csv):
    waveform = read_waveform(mars_csv)
    assert len(waveform) > 1000
    assert waveform.sampling_rate == pytest.approx(20.0, rel=0.01)


def test_reads_apollo_column_names(tmp_path):
    path = tmp_path / "lunar.csv"
    path.write_text(
        "time_abs,time_rel(sec),velocity(m/s)\n"
        "1970-01-19T20:25:00,0.0,1e-10\n"
        "1970-01-19T20:25:00,0.15,2e-10\n"
        "1970-01-19T20:25:00,0.30,3e-10\n"
    )
    waveform = read_csv_waveform(path)
    assert len(waveform) == 3
    assert waveform.sample_spacing == pytest.approx(0.15)


def test_mseed_and_csv_of_one_event_agree(mars_csv, mars_mseed):
    """The two exports of a record describe the same trace."""
    stem = mars_mseed.with_suffix("").name
    partner = mars_mseed.with_suffix(".csv")
    if not partner.exists():
        pytest.skip(f"no CSV counterpart for {stem}")

    from_csv = read_waveform(partner)
    from_mseed = read_waveform(mars_mseed)
    assert len(from_csv) == pytest.approx(len(from_mseed), rel=0.001)
    assert from_csv.sampling_rate == pytest.approx(from_mseed.sampling_rate, rel=0.01)


def test_rejects_unknown_extension(tmp_path):
    path = tmp_path / "trace.txt"
    path.write_text("nothing useful")
    with pytest.raises(WaveformError, match="unsupported file type"):
        read_waveform(path)


def test_rejects_missing_velocity_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("time_rel(sec),temperature\n0.0,1.0\n1.0,2.0\n")
    with pytest.raises(WaveformError, match="no velocity column"):
        read_csv_waveform(path)


def test_rejects_single_sample():
    with pytest.raises(WaveformError, match="at least two samples"):
        Waveform(time=np.array([0.0]), velocity=np.array([1.0]))


def test_rejects_mismatched_lengths():
    with pytest.raises(WaveformError, match="lengths differ"):
        Waveform(time=np.arange(5.0), velocity=np.arange(3.0))


def test_sample_spacing_survives_a_duplicated_timestamp():
    """A single bad timestamp must not distort the frequency axis."""
    time = np.arange(0.0, 10.0, 0.1)
    time[5] = time[4]  # a duplicate, as seen in some raw exports
    waveform = Waveform(time=time, velocity=np.zeros_like(time))
    assert waveform.sample_spacing == pytest.approx(0.1)


def test_after_trims_to_the_arrival(synthetic_quake):
    trimmed = synthetic_quake.after(600.0)
    assert trimmed.time[0] >= 600.0
    assert len(trimmed) < len(synthetic_quake)


def test_after_rejects_a_time_past_the_record(synthetic_quake):
    with pytest.raises(WaveformError, match="no usable samples"):
        synthetic_quake.after(99_999.0)
