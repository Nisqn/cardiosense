"""CardioSense - Step 1: """
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import wfdb

DATA_DIR = Path("data/mitdb")
FIG_DIR = Path("figures")
RECORD = "208"
SECONDS_TO_PLOT = 10


if not DATA_DIR.exists():
    print("Downloading MIT-BIH from PhysioNet...")
    wfdb.dl_database("mitdb", str(DATA_DIR))

# 2. Read the signal
record = wfdb.rdrecord(str(DATA_DIR / RECORD))
ann = wfdb.rdann(str(DATA_DIR / RECORD), "atr")

fs = record.fs                    # sampling frequency
signal = record.p_signal[:, 0]    # first lead (MLII), in mV

print(f"Record {RECORD}: {len(signal)} samples, fs = {fs} Hz, "
      f"duration = {len(signal) / fs / 60:.1f} min")
print(f"Leads: {record.sig_name}")
print(f"Annotation counts: {Counter(ann.symbol)}")

# 3. Plot 
n = SECONDS_TO_PLOT * fs
t = [i / fs for i in range(n)]

plt.figure(figsize=(14, 4))
plt.plot(t, signal[:n], linewidth=0.8)
for sample, symbol in zip(ann.sample, ann.symbol):
    if sample < n:
        plt.plot(sample / fs, signal[sample], "ro", markersize=4)
        plt.text(sample / fs, signal[sample] + 0.15, symbol,
                 ha="center", color="red")
plt.title(f"MIT-BIH record {RECORD} (lead {record.sig_name[0]}), first {SECONDS_TO_PLOT} s")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid(alpha=0.3)
plt.tight_layout()

FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / f"record{RECORD}.png", dpi=150)
plt.show()