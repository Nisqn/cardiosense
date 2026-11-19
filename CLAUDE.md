# CardioSense

ECG arrhythmia classification on MIT-BIH that stays honest under noise. Hackathon project (Devpost).

- **Deadline:** 26 Dec 2026. **Target submission:** 23 Dec 2026 (3 days buffer).
- **Started:** 19 Nov 2026. Commit often; the git history must show the work started after 1 Nov.

## What the project is

Pipeline: raw ECG → bandpass filter → R-peak detection → beat features → LightGBM classifier → abstention (signal quality + confidence) → SHAP explanations → Streamlit dashboard.

Classes (AAMI): **N** (normal), **S** (supraventricular ectopic), **V** (ventricular ectopic). F and Q are excluded.

The four things that make it stand out:
1. **Controlled noise experiments (core idea, never cut).** Add real NSTDB noise (baseline wander `bw`, muscle artifact `ma`, electrode motion `em`) at SNR 24, 18, 12, 6, 0, -6 dB. Plot F1 vs SNR per noise type.
2. **Abstention.** Signal quality index (e.g. kurtosis, agreement between two detectors) plus model confidence. Unreliable beats get no label. Show coverage vs accuracy.
3. **Readable explanations.** SHAP on named features, e.g. "RR interval 30% shorter than local average → S beat", drawn on the ECG.
4. **Live dashboard.** Record picker, noise slider, coloured beats, confidence, explanation panel. Deployed on Streamlit Community Cloud. This is the demo video.

## Hard rules for evaluation

- **Inter-patient split only.** de Chazal DS1 for training, DS2 for testing. Never let beats from the same record appear in both. A random beat-level split gives ~99% accuracy that means nothing.
- Report **per-class sensitivity, PPV and F1** plus the confusion matrix. Do not lead with accuracy.
- Paced records (102, 104, 107, 217) are excluded, as in de Chazal.
- Any tuning (thresholds, hyperparameters) uses DS1 only (e.g. grouped CV by record). DS2 is touched only for final numbers.
- Noise is added to the test signal before filtering/detection, so detection errors propagate into classification. Use a fixed random seed.

de Chazal split, for reference:
- DS1: 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
- DS2: 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

AAMI mapping (MIT-BIH symbol → class): N, L, R, e, j → N; A, a, J, S → S; V, E → V. Drop everything else (F, Q, paced, non-beat annotations).

## Tech stack

Python, wfdb, NumPy/SciPy, scikit-learn, LightGBM, SHAP, Streamlit, Plotly, matplotlib. Virtual env lives in `.venv/`; dependencies in `requirements.txt`. Platform is Windows.

- Data comes from PhysioNet via `wfdb` (`mitdb`, `nstdb`). Cache downloads in a local `data/` folder and keep it out of git.
- Sampling rate for MIT-BIH and NSTDB is 360 Hz. Don't hardcode it in more places than one.

## Roadmap

| Week | Dates | Focus | Done when |
|---|---|---|---|
| 1 | 19–25 Nov | Data loading, plotting, 0.5–40 Hz bandpass, Pan-Tompkins R-peak detection, detector eval (Se/PPV over all records) | Detector >99% sensitivity on clean records |
| 2 | 26 Nov–2 Dec | Beat segmentation, N/S/V mapping, features (pre/post RR, local/global avg RR, QRS width, amplitudes, morphology samples or wavelets), DS1/DS2, LightGBM with class weights | Raw record → predictions with honest metrics |
| 3 | 3–9 Dec | NSTDB noise at 6 SNR levels, re-run detection + classification, F1 vs SNR, SQI + confidence rejection, coverage vs accuracy | Key result figure exists |
| 4 | 10–16 Dec | SHAP per beat and global, Streamlit app, deploy | A stranger can use the link |
| 5 | 17–23 Dec | README (incl. limitations, "not a diagnostic tool"), unit tests for filter and detector, demo video <3 min, Devpost write-up | Submitted 23 Dec |

Stretch (only if ahead): small 1D CNN vs LightGBM; test on INCART for generalisation.

**Scope rule:** if a week slips, cut stretch items first, then dashboard polish. Never cut the noise experiment.

## How to work in this repo

- Keep the code small and readable. This is a 5-week solo project; prefer plain functions over frameworks.
- Put reusable logic in a package (e.g. `cardiosense/`), not in notebooks. Notebooks or scripts can call it for figures.
- Detection uses a matching tolerance window (standard: 150 ms) when comparing predicted R-peaks to annotations.
- Save result figures to a `figures/` folder; they go into the README and Devpost write-up.
- Only commit when asked. Don't commit `.venv/`, downloaded data, or large model files.
- When reporting results, state which split and which records were used.
- The user's DSP background is strong; explain ML choices more than filter theory.
