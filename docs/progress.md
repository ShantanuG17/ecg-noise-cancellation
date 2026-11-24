# Progress log

## 2026-11-22: Plan (written before any code)
Track B: adaptive ECG noise cancellation (Thakor & Zhu, 1991).
Baseline: LMS. Modification: NLMS (step normalized by reference energy).
Environment: conda (ecg-nc), Python 3.11, Windows 11.

### Pre-declared decisions
- Hypothesis: LMS stability/speed depends on reference power, so a mu tuned at
  one reference scale is mismatched at others. NLMS should be more robust
  across scales.
- Predicted help: reference scale differs from the tuning scale.
- Predicted hurt: near-silent reference (eps dominates) and slightly higher
  steady-state misadjustment at the nominal scale.
- Primary metric: SNR improvement (dB) = SNR(e) - SNR(d), after 1 s warm-up.
  Larger is better. Reported as the paired NLMS - LMS difference per recording.
- Seeds: dev 0-19, validation 100-109, test 1000-1019.
- Tuning: validation only, at reference scale 1.0, equal grid size per method.
- Test: reference scales {0.5, 1, 2}, 20 paired recordings each, plus stress
  scales, changing pathway, and leakage case (r <- r + 0.2 s).
- Uncertainty: 95% CI over recordings (paired).
- Ablation: to be decided in the experiments step.

## 2026-11-22: Data and metrics
- data.py: guide generator + optional pathway change (switch at change_at s,
  second pathway [-0.20, 0.30, -0.10]) and leakage (r <- r + leak*s).
- metrics.py: MSE, SNR, primary metric SNR improvement after warm-up.
  Non-finite outputs give NaN so divergence is recorded, not hidden.
- tests/test_data_metrics.py: shapes, determinism, split disjointness,
  scale/pathway/leakage behaviour, hand-computed metric values.

## 2026-11-22: LMS baseline
- common.py: delay_matrix (shared by baseline and modification, so both get identical inputs).
- baseline.py: LMS exactly as in the guide (w0 = 0, residual before update,
  half-squared-error convention). Divergence is left visible as inf/nan.
- tests: guide hand example (e=[1,0.5], w=0.75), zero reference, mu=0, M > N,
  no input mutation, argument validation, SNR gain on synthetic data,
  divergence at mu far above the stability limit.
  
## 2026-11-22: Dev observations and tuning plan (before experiments)

- Observation (dev seed 0 only, before experiments), SNR improvement in dB:
  scale 0.1: NLMS mu=0.5 -8.8, NLMS mu=0.05 2.2, LMS mu=0.02 1.6
  scale 1:   NLMS mu=0.5 8.2,  NLMS mu=0.05 13.2, LMS mu=0.02 14.5
  scale 10:  NLMS mu=0.5 27.8, NLMS mu=0.05 14.4, LMS mu=0.02 diverged (NaN)
  Findings: (a) LMS diverges at high scale, NLMS does not (supports hypothesis);
  (b) best NLMS mu still depends on scale because the ECG acts as adaptation
  noise independent of reference scale; (c) my prediction that LMS would barely
  adapt at scale 0.1 was wrong (it gains 1.6 dB). mu must be tuned on validation
  data for both methods, not chosen by hand. Test threshold at scale 0.1 removed
  (unchecked assumption of mine).

### Pre-declared tuning decisions
- Grids (validation, scale 1 only): M in {4, 8, 16, 32}. LMS mu log-spaced
  1e-4 to 1e-1; NLMS mu log-spaced 1e-3 to 1. 8 mu values each (32 configs
  per method). Best = highest mean SNR improvement on the 10 validation recordings.
- Divergence: a diverged run counts as a failure, reported as a count per
  method and condition, never dropped silently. Paired differences are
  reported on recordings where both ran, plus the failure counts.
- Metric caveat: SNR improvement depends on interference strength, so compare
  methods within a scale, not across scales.

## 2026-11-24: Experiment design (declared before the full run)
- Main comparison: LMS and NLMS tuned on validation (seeds 100-109) at
  reference scale 1.0 only, then tested on seeds 1000-1019 at every condition.
  This is the scale-mismatch scenario in the hypothesis.
- Ablation: both methods retuned on validation at each condition
  (lms_retuned, nlms_retuned). Shows how much of any NLMS advantage is just
  "no retuning needed".
- Conditions: scale 0.5, 1, 2 (main); scale 0.1 and 10 (stress);
  changing pathway at 6 s; leakage r <- r + 0.2 s.
- Primary metric: SNR improvement (dB) after 1 s warm-up, paired by recording.
- Divergence counted as failure and reported, never dropped silently.
- Smoke run (tiny grid) showed NLMS picking the top of its mu grid (1.0) at
  scale 10. Grid is NOT changed because of this; if it recurs in the full
  run it is reported as a limitation.  

## AI assistance
Used Claude to plan the project structure and experiment protocol.