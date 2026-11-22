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

## AI assistance
Used Claude to plan the project structure and experiment protocol.