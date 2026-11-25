# Adaptive ECG noise cancellation: LMS vs NLMS

EEP55C36 final project, Track B. Based on N. V. Thakor and Y.-S. Zhu (1991),
IEEE Trans. Biomed. Eng. 38(8), 785-794, doi:10.1109/10.83591.
Synthetic data only. No clinical claims.

## What this does
Reference-noise cancellation with an adaptive FIR filter. A reference channel
resembling the interference predicts it, and the residual is the cleaned ECG.
Only the LMS part of the paper is reproduced (no arrhythmia detection).

## What changed (baseline vs modification)
- Baseline, `src/baseline.py`: LMS, `w <- w + mu * e * x`.
- Modification, `src/modified.py`: NLMS, `w <- w + mu * e * x / (eps + x^T x)`.
  This is the only line that differs. Hypothesis: normalizing by reference
  energy makes the step robust to reference scale.
- Baseline frozen at git tag `baseline-frozen` before the change was written.

## Main findings (20 paired test recordings per condition, 95% CI)
Primary metric: SNR improvement (dB) after 1 s warm-up, NLMS minus LMS.
- Tuned at scale 1 only (mismatch): LMS better at scales 0.5, 1, 0.1 and
  leakage (-0.5 to -2.5 dB). NLMS better at scale 2 (+3.8 dB) and with a
  changing pathway (+1.3 dB). At scale 10 LMS diverged on 20/20 recordings and
  NLMS on 0/20.
- Both retuned per condition (ablation): differences shrink to within about
  +/-1 dB. NLMS ahead at scales 2 and 10, changing pathway and leakage; LMS
  ahead at scale 1; scales 0.5 and 0.1 indistinguishable.
- Runtime: NLMS is about 1.3x LMS per recording (pure Python loop).
- Mixed result: much of NLMS's advantage is not needing retuning for a new
  reference scale. See `results/final/*.csv` and `report/`.

## Repository layout
- `src/data.py` generator (guide's `make_ecg` plus pathway change and leakage)
- `src/baseline.py`, `src/modified.py`, `src/common.py` algorithms, shared delay matrix
- `src/metrics.py` MSE, SNR, SNR improvement (NaN if diverged)
- `src/experiment.py` tuning and paired test grid; `src/analysis.py` paired CIs
- `run_experiments.py` data -> tuning -> test grid -> CSV
- `analyze_results.py` tables and figures from saved CSV
- `configs/smoke.json`, `configs/final.json`; `tests/`; `results/`; `report/`; `docs/progress.md`

## Tested environment
Windows 11 Home (10.0.26200), AMD Ryzen 7 260 (8 cores / 16 threads), 16 GB RAM,
Python 3.11.17. Pinned versions in `requirements.txt` (also `environment.yml`).
Exact versions used for a run are saved in `results/<run>/environment.json`.

## Install
```
git clone https://github.com/ShantanuG17/ecg-noise-cancellation
cd ecg-noise-cancellation
git checkout v1.0-submission
conda env create -f environment.yml
conda activate ecg-nc
```
or `python -m venv .venv`, activate it, then `python -m pip install -r requirements.txt`.

## Quick run (smoke, about 6 s on the machine above)
```
python run_experiments.py --config configs/smoke.json
```
Writes 36 rows to `results/smoke/` (results.csv, tuned.json, environment.json,
config_used.json). Directories are created automatically.

## Run checks (about 5 s, 33 tests)
```
python -m pytest -q
```

## Reproduce final results (about 2 min, under 1 GB RAM)
```
python run_experiments.py --config configs/final.json
python analyze_results.py --results results/final
```
Expect 560 rows in `results/final/results.csv`, tables
(`paired_main_snr.csv`, `paired_ablation_snr.csv`, `paired_qrs_error.csv`,
`runtime_ratio.csv`, `divergence_counts.csv`) and four figures in
`results/final/figures/`: `fig_overall_paired.png`, `fig_example.png`,
`fig_stress_example.png`, `fig_stress_summary.png`.
Results are deterministic (seeded); only `runtime_s` varies by machine.

## Data, seeds and protocol
- Synthetic recordings, 12 s at 250 Hz. One seed = one whole recording.
- Seeds: development 0-19, validation 100-109, test 1000-1019. Tests never use clean test truth to choose settings.
- Tuning on validation only: M in {4, 8, 16, 32}, 8 log-spaced mu values per
  method (LMS 1e-4..1e-1, NLMS 1e-3..1), same budget for both.
- Main comparison tuned at scale 1.0, tested at scales 0.5, 1, 2; stress
  scales 0.1 and 10; pathway change at 6 s; leakage r <- r + 0.2 s.
- Uncertainty: 95% t-interval (bootstrap also saved) over recordings, paired by seed.
- Timing: warm-up run, then median of 3 timed runs per recording; application
  time only (tuning excluded; NLMS energy computation included).
- Diverged runs: NaN SNR, excluded from the paired difference and counted in `divergence_counts.csv`.

## Limitations
- One synthetic generator; results do not transfer to real ECG or clinical use.
- Grid edges: M=4 (smallest) was chosen in most cases; LMS hit its top mu (0.1) in a few retuned cases. Grids were not changed after seeing results.
- Differences of a few tenths of a dB on 20 recordings are small.
- NLMS's best mu still varies with scale (the ECG acts as adaptation noise).
- Runtime ratio reflects a pure-Python loop, not an optimized implementation.
- Assumption that can fail: the reference must be uncorrelated with the ECG. The leakage condition tests this.

## Citations
- Thakor and Zhu (1991), doi:10.1109/10.83591.
- S. Haykin, Adaptive Filter Theory (NLMS), implemented independently here.
- Course math guide (data generator, metrics), EEP55C36, Dr. D. Ramsook.
- Libraries: NumPy, SciPy, pandas, Matplotlib, pytest.

## AI assistance and attribution declaration
I used Claude (Anthropic) to help plan the project, draft code scaffolding, tests
and analysis scripts, and to discuss results. I ran, checked and can explain all
code and claims. No external adaptive-filter implementation was used.
