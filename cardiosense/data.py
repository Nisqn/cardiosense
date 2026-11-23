"""Loading MIT-BIH records.

Every other part of CardioSense gets its data through load_record(), so the
details of the wfdb library live in this one file only.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

# data/mitdb inside the repo, wherever you run the code from
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mitdb"


@dataclass
class ECGRecord:
    """One recording: the signal of one lead plus the cardiologists' labels."""
    record_id: str
    signal: np.ndarray        # voltage in mV, one value per sample
    fs: int                   # sampling frequency in Hz (360 for MIT-BIH)
    lead: str                 # lead name, e.g. "MLII"
    ann_samples: np.ndarray   # sample index of each annotation
    ann_symbols: np.ndarray   # label of each annotation ("N", "V", "+", ...)

    @property
    def time(self) -> np.ndarray:
        """Time axis in seconds, same length as signal."""
        return np.arange(len(self.signal)) / self.fs


def load_record(record_id: str, lead: int = 0) -> ECGRecord:
    """Load one MIT-BIH record. lead=0 is MLII in almost every record."""
    path = str(DATA_DIR / record_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    return ECGRecord(
        record_id=record_id,
        signal=rec.p_signal[:, lead],
        fs=int(rec.fs),
        lead=rec.sig_name[lead],
        ann_samples=np.asarray(ann.sample),
        ann_symbols=np.asarray(ann.symbol),
    )
