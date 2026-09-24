# Factor information-timing audit — FAIL

Audited panel: `/cloud/factor_synthesis/data/alphagat_64f_fixed_2018_2026_t1t2_vwap`  
Audited source: `/cloud/alphagat_stage2/scripts/build_server_64f_fixed_h5.py`  
Audit date: 2026-09-17

## Conclusion

**FAIL.** The raw per-date factor-value alignment and daily cross-sectional
preprocessing are point-in-time in the inspected panel builder.  However, the
actual 64-factor *set*, its order, and each factor's direction were selected
using an IC summary containing 2018–2026, including the prohibited 2022 test
period and later years.  Thus the features presented at dates in 2018–2021
embed future label information through global selection and sign choice.

No Factor Mean baseline, Experiment A, or other training/evaluation was run
after this finding.

## Evidence of the future-selection failure

1. `/cloud/factor_synthesis/factors/selected_64_input/selected_factors_abs_ic_gt_0.01.csv`
   has SHA-256
   `c016bf724913cd83618f69f53aa1faf8094dfe28e30fe3e67fd140a7ca87f662`.
2. That SHA-256 exactly equals the Stage-1 full-sample file at
   `/cloud/factor_synthesis/outputs/stage1_factor_correlation_absic_gt001_20260910/selected_factors_abs_ic_gt_0.01.csv`.
3. Its rows state `n_years=9` and `total_observations=2103`; the accompanying
   Stage-1 manifest explicitly lists 2018, 2019, 2020, 2021, 2022, 2023,
   2024, 2025, and 2026.
4. The builder's `_read_factor_selection` function reads this IC metadata
   (lines 71–84), filters it by the correlation manifest, assigns the global
   `direction`, sorts descending by global `abs_ic`, and retains `.head(64)`.
5. The builder then multiplies every raw historical factor by that direction
   at line 184.  Therefore both factor inclusion and factor sign for 2018–2021
   use information calculated with later labels.

This is a full-sample factor-selection/direction look-ahead. It is not a
mere model-selection issue, and it invalidates the requested out-of-sample
isolation for a 2021 validation experiment.

## Confirmed point-in-time properties of the panel builder

- Factor values are read at the matching decision date: `desired` is the
  pricing-date sequence (lines 119–125), and `factor_rows` is looked up using
  the same date (lines 173–183). The inspected code does not index factor
  values at `t+1` or later.
- The `t+1` and `t+2` rows at lines 126–156 build return labels only. They are
  separate from the factor-value read at lines 172–206.
- Imputation, winsorisation, mean and standard deviation are all calculated
  within the loop over one date `t` (lines 188–205). No temporal aggregate,
  `shift`, or centred rolling operation appears in the inspected panel builder.
- This panel itself is full-market. CSI500 membership is later supplied by a
  dated membership NPZ in E2EAI, so the inspected panel builder does not use
  a future index constituent list to form its daily index universe.

## Items that cannot be fully proven from the inspected code

- The individual raw factor H5 files are precomputed inputs. Their original
  formula implementations, any rolling-window convention, and financial-data
  publication timestamps are not contained in the panel builder or the H5
  metadata inspected here.
- Consequently, even after fixing full-sample factor selection, a separate
  upstream audit is required to prove each raw factor uses only data publicly
  available by date `t`.

## Required remediation before baseline or Experiment A

For this 2018–2021 training/validation study, create a frozen 64-factor
manifest and direction file using only information available before the first
evaluation interval. At minimum, the manifest/directions must exclude 2021
validation and all 2022+ labels. Rebuild or create a new panel using that
manifest, then repeat the execution-return verification and this audit. Do
not overwrite the present panel.

## 2022 isolation

No 2022 factor IC, score, prediction, portfolio weight, return, Sharpe, or
model result was computed in this audit. The presence of 2022 in the audited
upstream selection metadata is the reason for the FAIL finding.
