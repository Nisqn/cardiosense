"""Checks that the band-pass filter keeps what it should and removes what it should.

Run from the repo root with:  pytest
"""
import numpy as np

from cardiosense.filters import bandpass

FS = 360
t = np.arange(0, 20, 1 / FS)            # 20 s of samples


def gain(freq_hz: float) -> float:
    """Pass a pure sine through the filter and return output/input amplitude."""
    x = np.sin(2 * np.pi * freq_hz * t)
    y = bandpass(x, FS)
    mid = slice(len(t) // 4, 3 * len(t) // 4)   # ignore start/end transients
    return np.std(y[mid]) / np.std(x[mid])


def test_passband_kept():
    for f in (1, 5, 10, 20):             # heart-relevant frequencies
        assert gain(f) > 0.95


def test_baseline_wander_removed():
    assert gain(0.05) < 0.05             # slow drift


def test_mains_and_muscle_noise_removed():
    assert gain(60) < 0.1                # US mains frequency
    assert gain(100) < 0.01


def test_dc_offset_removed():
    y = bandpass(np.full_like(t, -0.35), FS)    # constant -0.35 mV like record 100
    assert abs(y[len(t) // 2]) < 1e-3


def test_no_peak_shift():
    """Zero-phase: a sharp spike must come out at the same sample."""
    x = np.zeros_like(t)
    x[3600] = 1.0
    assert np.argmax(bandpass(x, FS)) == 3600
