# CAFA5 check-in: Preliminary real-data internal validation

Validation: **preliminary exact-duplicate-grouped validation**.

## Reproduce

```sh
/Users/dylan/compmed_project/.venv/bin/python pilot.py --data-dir data/cafa5 --max-proteins 12000 --output outputs/real_pilot
```

## Data and method

- Source: third-party Zenodo mirror of Kaggle CAFA5 data.
- Raw sequence records: 142,246; unique sequence accessions: 142,246; joined: 142,246.
- Eligible non-root MF proteins: 78,637; pilot: 12,000; outside pilot: 66,637.
- Train / validation: 9,600 / 2,400 (20.0% validation); seed 42.
- Retained terms: 121 with at least 100 training positives after is_a propagation.
- Split checks passed: zero shared accessions, groups, or exact sequences.
- Evaluation: all 2,400 validation proteins for both methods; 2,390 have retained truth for recall.
- Both use the same ordered labels and proteins, with max-score propagation to retained ancestors through the full ontology.
- Logistic regression: 421 sequence features, training-only StandardScaler, liblinear, C=1, max_iter=1000, one CPU thread.

## Preliminary internal results

| Method | Micro AP | Restricted MF F-max | Threshold | Fit (s) | Inference (s) |
|---|---:|---:|---:|---:|---:|
| training_frequency | 0.3139 | 0.5120 | 0.17 | 0.0005 | 0.0012 |
| logistic_regression | 0.3089 | 0.5476 | 0.38 | 105.1600 | 0.1260 |

Logistic regression micro AP minus frequency micro AP: -0.0050 on this holdout.

Total pipeline time: 112.00 s. Inference batch: 2,400 proteins.
Hardware: macOS-26.5.2-arm64-arm-64bit-Mach-O, arm64; Python 3.13.15; one CPU thread.
Fit time includes training feature extraction/scaling. Inference includes validation feature extraction, scaling, prediction, and hierarchy postprocessing; input file loading and evaluation are excluded. Timings are single-run measurements.

## Example output

Validation accession: `A0A023PGP0`. Unannotated does not mean biologically absent.

| GO term | Name | Logistic score | Annotated positive |
|---|---|---:|---:|
| GO:0005488 | binding | 0.9227 | 1 |
| GO:0005515 | protein binding | 0.8169 | 0 |
| GO:0000976 | transcription cis-regulatory region binding | 0.7261 | 1 |
| GO:0000977 | RNA polymerase II transcription regulatory region sequence-specific DNA binding | 0.7261 | 0 |
| GO:0001067 | transcription regulatory region nucleic acid binding | 0.7261 | 1 |
| GO:0003676 | nucleic acid binding | 0.7261 | 1 |
| GO:0003677 | DNA binding | 0.7261 | 1 |
| GO:0003690 | double-stranded DNA binding | 0.7261 | 1 |
| GO:0043565 | sequence-specific DNA binding | 0.7261 | 1 |
| GO:0097159 | organic cyclic compound binding | 0.7261 | 1 |

## Limits and next step

- Exact duplicates are grouped; close homologs may still cross the split. Similarity-grouped validation remains open.
- Scores concern frequent retained MF terms only; these are not official CAFA temporal benchmark scores.
- Zeros denote missing annotations. AP uses the observed-label convention; precision can penalize plausible but unannotated functions.
- F-max sweeps 101 thresholds from 0.00 to 1.00 inclusive (score >= threshold). Its maximizing threshold is descriptive for this holdout, not a tuned deployment threshold.
- 10 validation proteins have no retained truth: included in AP and precision; excluded from recall by definition.
- Nonconverged logistic terms: 0 (IDs in data_summary.json).
- Next: similarity-grouped validation, then a BLAST/DIAMOND homology baseline and grouped tuning.

- This is a seeded whole-group pilot subset, not a full-training-data experiment.

Data source: [third-party Zenodo mirror of Kaggle CAFA5 data](https://zenodo.org/records/10951709).
