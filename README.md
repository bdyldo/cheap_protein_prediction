# CAFA5 sequence-only check-in pilot

Compare a training-frequency baseline with one-vs-rest logistic regression on an internal Molecular Function holdout. This is a course progress experiment, not the official CAFA temporal evaluation.

## Setup

Python 3.13 was used for the recorded run.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Get data and run

```sh
python download_data.py --source auto
python pilot.py --data-dir data/cafa5 --max-proteins 12000 --output outputs/real_pilot
```

The downloader first tries the official `kagglehub.competition_download('cafa-5-protein-function-prediction')` route. If access fails, it uses a [third-party Zenodo mirror](https://zenodo.org/records/10951709), verifies the archive against its published MD5, and extracts only the three training inputs. The mirror is described by its depositor as Kaggle CAFA data; archive integrity is verified, but byte-for-byte equivalence with official Kaggle downloads is not independently established. File SHA-256 checksums and provenance are saved with the data and results. No temporal evaluation labels enter the pilot.

For the official download only, run `python download_data.py --source kaggle --output data/cafa5_official`. Kaggle may require authentication and accepting the competition rules in your own account. Alternatively, supply a directory containing `train_sequences.fasta`, `train_terms.tsv`, and `go-basic.obo`.

The 12,000-protein cap creates a manageable, reproducible pilot by selecting whole groups before the split. The exact count may differ slightly to preserve groups. Omit `--max-proteins` for all eligible proteins. Term selection always uses at least 100 positives in the training partition alone.

## What to open for the TA

- [Check-in results](outputs/real_pilot/CHECKIN_RESULTS.md): data counts, two-method results, timing, one example prediction, and limitations.
- [Data summary](outputs/real_pilot/data_summary.json): input hashes, actual headers, filtering counts, configuration, hardware, and versions.
- [Pipeline](pilot.py): loading, grouping, features, training, hierarchy postprocessing, and evaluation.
- [Demo notes](TA_DEMO.md): a short walkthrough and talking points.

Show the saved table and one protein's predictions with `python demo.py`; this does not rerun training. Use `--protein ACCESSION` to inspect another protein in the validation set.

## Method and limits

- Join FASTA accessions with `EntryID` annotations; accept bare accessions and UniProt `sp|...|...` / `tr|...|...` headers. Deduplicate protein–term pairs.
- Keep MF, resolve ontology alternative IDs, propagate only `is_a`, and exclude root `GO:0003674`, unknown terms, and obsolete terms. The supplied ontology version is recorded; current GO is not substituted.
- Default split: seed 42, approximately 80/20, whole exact-sequence groups. Zero shared accessions, groups, or exact sequences are asserted. Close homologs can still cross this split.
- Optional `--groups path/to/groups.csv` accepts `accession,group_id`. Provide assignments for all eligible proteins, keep exact duplicates together, and document how the groups were made. External clustering commands, tool versions, identity, and coverage settings should accompany those groups.
- Features: raw sequence length, 20 amino-acid frequencies, 400 adjacent pair frequencies. Nonstandard residues contribute to length, are excluded from composition denominators, and break pairs. Sequences with no valid pairs receive zero pair features.
- Scaling is fitted on training features. Each retained term uses `LogisticRegression(solver='liblinear', C=1, max_iter=1000, tol=1e-4, random_state=42)`; all-positive labels use a constant. Both methods propagate scores to retained ancestors through the full graph.
- Micro AP covers every validation protein–term pair. Restricted MF F-max sweeps 0.00–1.00 inclusive in steps of 0.01. Precision averages over proteins with predictions; recall averages over proteins with retained annotations. All validation proteins are scored by both methods. Zero-truth rows remain in AP and precision. The maximizing threshold is descriptive for this holdout.
- A zero denotes missing annotation, not demonstrated absence. Conventional observed-label metrics and binary training treat these entries as unlabeled negatives; interpret results accordingly.
- Training time includes feature extraction/scaling; inference time includes features, scaling, scoring, and hierarchy postprocessing on the same full validation batch. File loading and metric calculation are excluded from inference. One CPU thread is used. Single-run times are preliminary measurements.

## Outputs

Each output directory contains `CHECKIN_RESULTS.md`, `data_summary.json`, `split.csv`, `term_support.csv`, `metrics.csv`, `threshold_metrics.csv`, `example_predictions.csv`, `unmatched_ids.csv`, `excluded_proteins.csv`, and `validation_predictions.npz`.

The NPZ stores ordered accessions, GO IDs, targets, and both score matrices so results can be audited. Load with `numpy.load(path, allow_pickle=False)`. Data and generated outputs are excluded from version control by default.

## Verify without real data

```sh
python -m unittest -v
python pilot.py --smoke-test --output outputs/smoke
```

The smoke test creates explicitly synthetic sequences and labels, exercises the complete pipeline, and uses a test-only support threshold of 8. Its metrics are not evidence of protein prediction performance.

Next experiments: similarity-grouped validation, a BLAST/DIAMOND homology baseline, and grouped tuning before independent temporal evaluation.
