"""CardioSense - Step 2: band-pass filter a record and show what changed.

Run from the repo root:
    python -m scripts.filter_demo            (record 105, a noisy one)
    python -m scripts.filter_demo 100        (any other record)

Saves two figures in figures/:
    filter_response.png        what the filter does to each frequency
    filter_demo_<record>.png   raw vs filtered signal + spectrum before/after
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from cardiosense.data import load_record
from cardiosense.filters import HIGH_HZ, LOW_HZ, bandpass, bandpass_sos

RECORD = sys.argv[1] if len(sys.argv) > 1 else "105"
START_S, LENGTH_S = 60, 10          # which 10-second window to plot
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load + filter
rec = load_record(RECORD)
fs = rec.fs
raw = rec.signal
filtered = bandpass(raw, fs)

# ------------------------------------------------- figure 1: frequency response
# filtfilt applies the filter twice, so the effective gain is |H(f)|^2
sos = bandpass_sos(fs)
f, h = sps.sosfreqz(sos, worN=8192, fs=fs)
gain_db = 20 * np.log10(np.maximum(np.abs(h) ** 2, 1e-12))

plt.figure(figsize=(9, 4))
plt.semilogx(f[1:], gain_db[1:])
plt.axvspan(LOW_HZ, HIGH_HZ, color="green", alpha=0.08, label="kept band")
plt.axvline(60, color="red", ls="--", lw=1, label="60 Hz mains")
plt.ylim(-80, 5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.title(f"Band-pass {LOW_HZ}-{HIGH_HZ} Hz, Butterworth, zero-phase")
plt.grid(which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "filter_response.png", dpi=150)

# ----------------------------------------- figure 2: signal and spectrum before/after
a, b = START_S * fs, (START_S + LENGTH_S) * fs
t = rec.time[a:b]
in_window = (rec.ann_samples >= a) & (rec.ann_samples < b)
beats = rec.ann_samples[in_window]
labels = rec.ann_symbols[in_window]

fig, ax = plt.subplots(3, 1, figsize=(14, 10))

ax[0].plot(t, raw[a:b], lw=0.8)
ax[0].set_title(f"Record {RECORD} ({rec.lead}) - raw")
ax[1].plot(t, filtered[a:b], lw=0.8, color="tab:orange")
ax[1].set_title("After band-pass filter")
for axis, x in ((ax[0], raw), (ax[1], filtered)):
    axis.plot(beats / fs, x[beats], "r.", ms=6)
    for s, lab in zip(beats, labels):
        axis.annotate(lab, (s / fs, x[s]), xytext=(0, 6),
                      textcoords="offset points", ha="center", color="red")
    axis.set_ylabel("mV")
    axis.grid(alpha=0.3)
ax[1].set_xlabel("Time (s)")

# Power spectral density (Welch): how much energy the signal has at each frequency
f_raw, p_raw = sps.welch(raw, fs, nperseg=4096)
f_flt, p_flt = sps.welch(filtered, fs, nperseg=4096)
ax[2].semilogy(f_raw, p_raw, label="raw", lw=1)
ax[2].semilogy(f_flt, p_flt, label="filtered", lw=1)
ax[2].axvspan(LOW_HZ, HIGH_HZ, color="green", alpha=0.08)
ax[2].set_xlim(0, fs / 2)
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("Power (mV²/Hz)")
ax[2].set_title("Spectrum of the whole record (Welch)")
ax[2].legend()
ax[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / f"filter_demo_{RECORD}.png", dpi=150)

# ------------------------------------------------------------------ numbers
def band_power(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return np.trapezoid(psd[m], freqs[m])

print(f"Record {RECORD}: {len(raw) / fs / 60:.1f} min, lead {rec.lead}, fs = {fs} Hz")
print(f"Mean level:  raw {raw.mean():+.3f} mV  ->  filtered {filtered.mean():+.3f} mV")
for name, lo, hi in (("baseline wander  < 0.5 Hz", 0, LOW_HZ),
                     ("ECG band     0.5-40 Hz", LOW_HZ, HIGH_HZ),
                     ("noise          > 40 Hz", HIGH_HZ, fs / 2)):
    before = band_power(f_raw, p_raw, lo, hi)
    after = band_power(f_flt, p_flt, lo, hi)
    print(f"  {name}: {100 * (1 - after / before):5.1f}% of the power removed")
print(f"Figures saved in {FIG_DIR}/")
plt.show()
