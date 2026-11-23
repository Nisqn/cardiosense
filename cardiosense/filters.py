"""ECG pre-processing filters

Band-pass 0.5-40 Hz:
  - below 0.5 Hz: baseline wander (breathing, electrode drift)  -> removed
  - 0.5-40 Hz:    P wave, QRS complex, T wave                   -> kept
  - above 40 Hz:  muscle noise, mains interference (60 Hz in the US) -> removed

We use a Butterworth filter (flat passband, no ripple), stored as second-order
sections (SOS) because a 0.5 Hz cutoff at fs = 360 Hz puts the poles very close
to z = 1, where the plain b/a form becomes numerically unstable.

sosfiltfilt runs the filter forwards and then backwards. The phase shifts cancel
(zero phase), so the R peaks stay exactly where the annotations say they are.
The price: it needs the whole signal in advance (non-causal). Fine for recorded
data; a live sensor would need a causal filter instead.
"""
import numpy as np
from scipy import signal as sps

LOW_HZ = 0.5
HIGH_HZ = 40.0
ORDER = 4


def bandpass_sos(fs: float, low: float = LOW_HZ, high: float = HIGH_HZ,
                 order: int = ORDER) -> np.ndarray:
    """Design the Butterworth band-pass filter and return its SOS coefficients."""
    return sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def bandpass(x: np.ndarray, fs: float, low: float = LOW_HZ, high: float = HIGH_HZ,
             order: int = ORDER) -> np.ndarray:
    """Zero-phase band-pass filter of signal x."""
    return sps.sosfiltfilt(bandpass_sos(fs, low, high, order), x)
