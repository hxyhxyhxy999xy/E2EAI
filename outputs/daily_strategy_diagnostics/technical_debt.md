# Technical debt and research-validity notes

## P0 — full-sample factor selection and direction (blocks this experiment)

The fixed 64-factor panel was selected and direction-aligned from a nine-year
2018–2026 IC summary. This uses validation/test-era labels to choose the
feature set and signs. It blocks Factor Mean and Experiment A for the intended
2018–2020 train / 2021 validation design. A new immutable, pre-validation
manifest and direction file are required; the existing panel must not be
overwritten.

## P2 — label-builder invalid-value storage semantics (does not block this audit)

The current panel-builder source contains a branch that writes `0.0` for an
invalid VWAP label, while the actual server file reused as daily execution
return was directly verified against raw HDF5 and has matching numerical and
NaN/finite semantics over 2018–2021. The existing verified file can therefore
be used. If labels are regenerated in future, reconcile this storage rule and
repeat the full mask and numeric verification before use.
