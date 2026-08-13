"""Arrival-time detection with the classic STA/LTA trigger.

STA/LTA slides a short and a long window across the trace and compares their
average power. A quake makes the short window spike while the long window
still reflects background noise, so the ratio crosses a threshold at the
onset of the event.

We use it only to find *where* the event starts; the classification itself
runs on the spectrum of everything after that point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cosmoquake.waveform import Waveform

#: Short-term window length in seconds.
DEFAULT_STA_SECONDS = 120.0

#: Long-term window length in seconds.
DEFAULT_LTA_SECONDS = 600.0

#: Ratio at which a trigger turns on.
DEFAULT_THRESHOLD_ON = 4.0

#: Ratio at which a trigger turns off again.
DEFAULT_THRESHOLD_OFF = 1.5


@dataclass(frozen=True)
class DetectionResult:
    """Where an event was found in a trace.

    Attributes:
        arrival_time: Onset of the first trigger, in seconds relative to the
            start of the record.
        end_time: Time the first trigger switched off, or ``None`` if it was
            still active when the record ended.
        triggered: ``False`` when no window crossed the threshold and the
            arrival time fell back to the start of the record.
        n_triggers: How many separate triggers the trace produced.
    """

    arrival_time: float
    end_time: float | None
    triggered: bool
    n_triggers: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the detection."""
        return {
            "arrival_time": self.arrival_time,
            "end_time": self.end_time,
            "triggered": self.triggered,
            "n_triggers": self.n_triggers,
        }


def characteristic_function(
    waveform: Waveform,
    sta_seconds: float = DEFAULT_STA_SECONDS,
    lta_seconds: float = DEFAULT_LTA_SECONDS,
) -> np.ndarray:
    """Return the STA/LTA ratio at every sample of the trace.

    Window lengths are clamped so short records still produce a usable
    function instead of raising: a ten-minute Mars segment cannot host a
    ten-minute long-term window.
    """
    from obspy.signal.trigger import classic_sta_lta

    n_samples = len(waveform)
    rate = waveform.sampling_rate

    # Keep both windows inside the record, and the short one strictly
    # shorter than the long one, or the ratio is meaningless.
    lta_samples = min(int(lta_seconds * rate), max(n_samples // 2, 2))
    sta_samples = min(int(sta_seconds * rate), max(lta_samples // 2, 1))
    sta_samples = max(sta_samples, 1)
    lta_samples = max(lta_samples, sta_samples + 1)

    return classic_sta_lta(
        np.asarray(waveform.velocity, dtype=float), sta_samples, lta_samples
    )


def detect_arrival(
    waveform: Waveform,
    sta_seconds: float = DEFAULT_STA_SECONDS,
    lta_seconds: float = DEFAULT_LTA_SECONDS,
    threshold_on: float = DEFAULT_THRESHOLD_ON,
    threshold_off: float = DEFAULT_THRESHOLD_OFF,
) -> DetectionResult:
    """Locate the first seismic arrival in a trace.

    Args:
        waveform: The full record to scan.
        sta_seconds: Short-term window length.
        lta_seconds: Long-term window length.
        threshold_on: Ratio at which a trigger starts.
        threshold_off: Ratio at which it stops.

    Returns:
        A :class:`DetectionResult`. When nothing crosses the threshold the
        arrival falls back to the start of the record with ``triggered``
        set to ``False``, so callers can still classify the whole trace and
        report the lower confidence in that decision.
    """
    from obspy.signal.trigger import trigger_onset

    ratio = characteristic_function(waveform, sta_seconds, lta_seconds)
    triggers = np.asarray(trigger_onset(ratio, threshold_on, threshold_off))

    start = float(waveform.time[0])
    if triggers.size == 0:
        return DetectionResult(
            arrival_time=start, end_time=None, triggered=False, n_triggers=0
        )

    # trigger_onset returns *sample indices*, not seconds. Converting through
    # the sample spacing is what keeps this correct for InSight's 20 Hz data
    # as well as Apollo's 6.6 Hz.
    spacing = waveform.sample_spacing
    on_index, off_index = int(triggers[0][0]), int(triggers[0][1])
    arrival_time = start + on_index * spacing
    end_time = start + off_index * spacing
    if off_index >= len(waveform) - 1:
        end_time = None

    return DetectionResult(
        arrival_time=arrival_time,
        end_time=end_time,
        triggered=True,
        n_triggers=int(triggers.shape[0]),
    )
