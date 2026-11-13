# TA progress check: November 13, 2026

## Open these

1. [Actual run report](outputs/real_pilot/CHECKIN_RESULTS.md)
2. [Pipeline code](pilot.py)
3. [Dataset summary](outputs/real_pilot/data_summary.json)

From the project folder, show the saved experiment without retraining:

```sh
.venv/bin/python demo.py
```

## Four-minute walkthrough

**Question (30 seconds):** Can lightweight sequence-only models produce useful protein function predictions at low inference cost? This first experiment establishes the data pipeline, evaluation, and a simple comparator; the eventual efficiency comparison needs stronger baselines.

**Data and validation (60 seconds):** Downloaded a public third-party CAFA5 mirror after the official Kaggle route required authentication. Verified the mirror archive checksum and recorded file SHA-256 hashes. The ontology is the 2023-01-01 release. Joined all 142,246 sequence accessions to annotations; 78,637 have usable non-root MF labels. Selected a seeded 12,000-protein pilot, split into 9,600 training and 2,400 validation proteins. Exact duplicate sequences stay together. Retained 121 terms with at least 100 training positives after `is_a` propagation.

**Experiment and result (90 seconds):** Compare the training-frequency baseline with separate logistic classifiers using sequence length, 20 amino-acid frequencies, and 400 adjacent pair frequencies. Both methods use identical proteins and labels, plus the same hierarchy postprocessing.

| Method | Micro AP | Restricted MF F-max | Inference for 2,400 proteins |
|---|---:|---:|---:|
| Training frequency | 0.3139 | 0.5120 | 0.0012 s |
| Logistic regression | 0.3089 | 0.5476 | 0.1260 s |

The initial result is mixed: logistic regression increases F-max by about 0.036 absolute but decreases micro AP by about 0.005. These measure different aspects of prediction; F-max summarizes per-protein precision/recall at one threshold, while micro AP summarizes ranking over all protein–term pairs. There is no overall performance win to claim yet.

Logistic fitting, including training features and scaling, took about 105 seconds. The complete pipeline took 112 seconds. Inference includes validation features, scaling, prediction, and score postprocessing. These are single-run CPU measurements, not a definitive hardware benchmark.

**Example and next step (60 seconds):** Run `demo.py` to show one validation protein's top predicted GO terms alongside observed annotations. Then explain the limitations and next experiment.

## Say these limitations plainly

- This is preliminary exact-duplicate-grouped validation. Related proteins may still cross the split, so generalization estimates can be optimistic.
- The pilot covers frequent Molecular Function labels in a subset of training data. It is not the independent CAFA temporal benchmark or the official CAFA metric.
- Missing GO annotations are not confirmed negative functions. A predicted but unannotated function is not necessarily wrong.
- Each F-max threshold was selected descriptively on this holdout; it is not a validated deployment choice.
- The mirror's published archive checksum passed; its equivalence to an authenticated official Kaggle download has not been independently checked.

## Evidence of working code

Nine automated checks pass, including hand-calculated metric expectations, exact-duplicate grouping, handling of nonstandard residues, annotation deduplication, and ancestor score propagation through unretained terms. A separate synthetic smoke run exercises the complete pipeline, including constant all-positive labels. The real run has zero nonconverged classifiers.

## Next experiment

Replace exact-duplicate groups with sequence-similarity groups and add a BLAST/DIAMOND homology baseline. Keep the current pipeline as the reproducible reference before tuning additional models.

Useful TA discussion: Which similarity grouping settings and evaluation protocol would make the final accuracy–efficiency comparison convincing?

## Reproduce the recorded experiment

```sh
.venv/bin/python download_data.py --source auto
.venv/bin/python pilot.py --data-dir data/cafa5 --max-proteins 12000 --output outputs/real_pilot
```

Setup and full methodology: [README](README.md). Data source: [Zenodo CAFA protein dataset](https://zenodo.org/records/10951709); original source: [Kaggle CAFA5](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data).
