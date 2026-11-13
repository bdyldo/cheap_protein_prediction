"""Reproducible, sequence-only CAFA5 Molecular Function internal validation."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys
import time
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MF_ROOT = 'GO:0003674'
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
INPUT_FILES = ('train_sequences.fasta', 'train_terms.tsv', 'go-basic.obo')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def accession(header):
    token = header.split()[0]
    parts = token.split('|')
    return parts[1] if len(parts) >= 3 and parts[0] in {'sp', 'tr'} else token


def load_fasta(path):
    sequences, stats, examples = {}, Counter(), []
    current, chunks = None, []

    def save():
        if current is None:
            return
        stats['fasta_records'] += 1
        sequence = ''.join(chunks).upper()
        if current in sequences:
            if sequences[current] != sequence:
                raise ValueError(f'Conflicting sequences for accession {current}')
            stats['duplicate_accession_records'] += 1
        sequences[current] = sequence

    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('>'):
                save()
                if len(examples) < 3:
                    examples.append(line[1:])
                current, chunks = accession(line[1:]), []
            elif line:
                if current is None:
                    raise ValueError('Sequence encountered before first FASTA header')
                chunks.append(''.join(line.split()))
        save()
    stats['unique_sequence_accessions'] = len(sequences)
    stats['empty_sequence_accessions'] = sum(not s for s in sequences.values())
    return sequences, dict(stats), examples


def load_ontology(path):
    nodes, aliases, metadata = {}, {}, {}
    current = None

    def save():
        if current and 'id' in current:
            nodes[current['id']] = current.copy()

    with Path(path).open() as handle:
        for raw in handle:
            line = raw.strip()
            if line == '[Term]':
                save()
                current = {'parents': [], 'alt_ids': [], 'obsolete': False}
            elif line.startswith('['):
                save()
                current = None
            elif ': ' in line:
                key, value = line.split(': ', 1)
                if current is None:
                    if key in {'format-version', 'data-version', 'date'}:
                        metadata[key] = value
                elif key in {'id', 'name', 'namespace'}:
                    current[key] = value
                elif key == 'is_a':
                    current['parents'].append(value.split()[0])
                elif key == 'alt_id':
                    current['alt_ids'].append(value)
                elif key == 'is_obsolete':
                    current['obsolete'] = value == 'true'
        save()
    for go_id, node in nodes.items():
        for alt in node['alt_ids']:
            if alt in aliases and aliases[alt] != go_id:
                raise ValueError(f'Ambiguous ontology alternative ID: {alt}')
            aliases[alt] = go_id
    if MF_ROOT not in nodes:
        raise ValueError('Ontology does not contain the Molecular Function root')

    @lru_cache(maxsize=None)
    def ancestors(go_id):
        # All intermediates are traversed, even if later excluded by support.
        result = {go_id}
        for parent in nodes[go_id]['parents']:
            parent = aliases.get(parent, parent)
            if parent not in nodes:
                raise ValueError(f'Missing ontology parent: {parent}')
            result.update(ancestors(parent))
        return frozenset(result)

    return nodes, aliases, ancestors, metadata


def load_annotations(path, nodes, aliases, ancestors):
    labels, pairs, annotation_ids = defaultdict(set), set(), set()
    stats, excluded_terms, direct_terms = Counter(), Counter(), set()
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        columns = reader.fieldnames
        if not columns or not {'EntryID', 'term', 'aspect'}.issubset(columns):
            raise ValueError(f'Expected TSV columns EntryID, term, aspect; found {columns}')
        for row in reader:
            stats['annotation_rows_raw'] += 1
            protein, term, aspect = (row[k].strip() for k in ('EntryID', 'term', 'aspect'))
            if not protein or not term:
                raise ValueError(f'Empty protein/term in annotation row {reader.line_num}')
            annotation_ids.add(protein)
            term = aliases.get(term, term)
            key = (protein, term)
            if key in pairs:
                stats['duplicate_protein_term_rows'] += 1
                continue
            pairs.add(key)
            if aspect not in {'F', 'MF', 'MFO', 'molecular_function'}:
                stats['non_mf_pairs_excluded'] += 1
                continue
            node = nodes.get(term)
            if node is None or node['obsolete']:
                stats['unknown_or_obsolete_mf_pairs_excluded'] += 1
                excluded_terms[term] += 1
                continue
            if node.get('namespace') != 'molecular_function':
                raise ValueError(f'MF aspect disagrees with ontology namespace: {term}')
            direct_terms.add(term)
            propagated = {t for t in ancestors(term)
                          if t != MF_ROOT and not nodes[t]['obsolete']
                          and nodes[t].get('namespace') == 'molecular_function'}
            labels[protein].update(propagated)
    stats['unique_protein_term_pairs'] = len(pairs)
    stats['annotation_accessions'] = len(annotation_ids)
    stats['mf_direct_distinct_terms_in_annotations'] = len(direct_terms - {MF_ROOT})
    stats['excluded_mf_terms'] = dict(excluded_terms)
    return dict(labels), annotation_ids, dict(stats), columns


def make_groups(ids, sequences, groups_path=None):
    if groups_path:
        with Path(groups_path).open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'accession', 'group_id'}.issubset(reader.fieldnames or []):
                raise ValueError('Group CSV needs accession,group_id columns')
            groups = {}
            for row in reader:
                protein, group = row['accession'].strip(), row['group_id'].strip()
                if not protein or not group or (protein in groups and groups[protein] != group):
                    raise ValueError('Missing or conflicting group assignment')
                groups[protein] = group
        missing = set(ids) - groups.keys()
        if missing:
            raise ValueError(f'{len(missing)} eligible proteins lack a supplied group')
        groups = {p: groups[p] for p in ids}
    else:
        groups = {p: hashlib.sha256(sequences[p].encode()).hexdigest() for p in ids}
    sequence_groups = {}
    for protein in ids:
        sequence = sequences[protein]
        if sequence in sequence_groups and sequence_groups[sequence] != groups[protein]:
            raise ValueError('Exact duplicate sequences have different supplied groups')
        sequence_groups[sequence] = groups[protein]
    return groups


def grouped_split(ids, groups, seed, fraction=0.2, max_proteins=None):
    members = defaultdict(list)
    for protein in sorted(ids):
        members[groups[protein]].append(protein)
    keys = sorted(members)
    np.random.default_rng(seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError('Need at least two independent sequence groups')
    if max_proteins and len(ids) > max_proteins:
        sizes = np.cumsum([len(members[k]) for k in keys])
        count = max(2, int(np.argmin(np.abs(sizes - max_proteins))) + 1)
        keys = keys[:count]
    total = sum(len(members[k]) for k in keys)
    cumulative = np.cumsum([len(members[k]) for k in keys])
    boundary = int(np.argmin(np.abs(cumulative[:-1] - total * (1 - fraction)))) + 1
    train = sorted(p for k in keys[:boundary] for p in members[k])
    validation = sorted(p for k in keys[boundary:] for p in members[k])
    return train, validation


def assert_split(train, validation, groups, sequences):
    assert not set(train) & set(validation), 'Accession leakage'
    assert not {groups[p] for p in train} & {groups[p] for p in validation}, 'Group leakage'
    assert not {sequences[p] for p in train} & {sequences[p] for p in validation}, 'Exact-sequence leakage'


def featurize(sequences):
    """Raw length + valid-AA composition + valid *adjacent* pair composition.

    Nonstandard residues contribute to length but not frequency denominators.
    Invalid residues break pairs (AXC does not count AC). Empty denominators
    give all-zero composition vectors, with no NaNs.
    """
    features = np.zeros((len(sequences), 421), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        codes = np.fromiter((AA_INDEX.get(aa, -1) for aa in sequence), dtype=np.int16)
        valid = codes[codes >= 0]
        features[row, 0] = len(sequence)
        if len(valid):
            features[row, 1:21] = np.bincount(valid, minlength=20) / len(valid)
        if len(codes) > 1:
            mask = (codes[:-1] >= 0) & (codes[1:] >= 0)
            pairs = codes[:-1][mask] * 20 + codes[1:][mask]
            if len(pairs):
                features[row, 21:] = np.bincount(pairs, minlength=400) / len(pairs)
    return features


def target_matrix(ids, labels, terms):
    index = {term: column for column, term in enumerate(terms)}
    target = np.zeros((len(ids), len(terms)), dtype=np.uint8)
    for row, protein in enumerate(ids):
        for term in labels[protein]:
            if term in index:
                target[row, index[term]] = 1
    return target


def ancestor_columns(terms, ancestors):
    index = {term: column for column, term in enumerate(terms)}
    return [[index[a] for a in ancestors(term) if a in index and a != term] for term in terms]


def propagate_scores(scores, columns):
    # Each column reaches ALL retained ancestors directly, so ordering is immaterial.
    for child, parents in enumerate(columns):
        for parent in parents:
            np.maximum(scores[:, parent], scores[:, child], out=scores[:, parent])
    return scores


def evaluate(target, scores, thresholds):
    """Restricted-label internal metrics; zeros mean unobserved annotations.

    AP conventionally scores these zeros as unlabeled negatives; this is not
    evidence of biological absence. F-max follows the specified per-protein
    precision/recall averaging, rather than flattening into a micro F1.
    """
    if target.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and aligned with targets')
    known_counts = target.sum(axis=1)
    known = known_counts > 0
    if not known.any():
        raise ValueError('No validation proteins have a retained annotated term')
    micro_ap = float(average_precision_score(target.ravel(), scores.ravel()))
    curve = []
    for threshold in thresholds:
        predicted = scores >= threshold
        prediction_counts = predicted.sum(axis=1)
        covered = prediction_counts > 0
        true_positive = (predicted & target.astype(bool)).sum(axis=1)
        precision = float(np.mean(true_positive[covered] / prediction_counts[covered])) if covered.any() else 0.0
        recall = float(np.mean(true_positive[known] / known_counts[known]))
        fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        curve.append(dict(threshold=float(threshold), precision=precision,
                          recall=recall, fscore=fscore, predicted_proteins=int(covered.sum())))
    best = max(curve, key=lambda row: row['fscore'])
    return dict(micro_ap=micro_ap, restricted_mf_fmax=best['fscore'],
                fmax_threshold=best['threshold'], fmax_precision=best['precision'],
                fmax_recall=best['recall']), curve


def make_fixture(directory):
    """Synthetic pipeline checks only; deliberately unrelated to real GO biology."""
    directory.mkdir(parents=True, exist_ok=True)
    ontology = ['format-version: 1.2', 'data-version: synthetic-smoke-test']
    for term, name, parent in [(MF_ROOT, 'molecular_function', None),
                               ('GO:9000001', 'synthetic shared parent', MF_ROOT),
                               ('GO:9000002', 'synthetic class A', 'GO:9000001'),
                               ('GO:9000003', 'synthetic class B', 'GO:9000001')]:
        ontology += ['', '[Term]', f'id: {term}', f'name: {name}', 'namespace: molecular_function']
        if parent:
            ontology.append(f'is_a: {parent} ! synthetic parent')
    (directory / 'go-basic.obo').write_text('\n'.join(ontology) + '\n')
    rng = np.random.default_rng(123)
    records, rows = [], []
    first_sequence = None
    for i in range(80):
        protein = f'SYNTHETIC_{i:03d}'
        alphabet = 'ACDEFGHIKL' if i % 2 == 0 else 'MNPQRSTVWY'
        sequence = ''.join(rng.choice(list(alphabet), size=60 + i))
        if i == 0:
            first_sequence = sequence
        if i == 79:
            sequence = first_sequence  # Duplicate accession group exercises leakage checks.
        records.append(f'>sp|{protein}|SYNTHETIC fixture only\n{sequence}\n')
        rows.append(dict(EntryID=protein, term=f'GO:900000{2 + i % 2}', aspect='F'))
    (directory / 'train_sequences.fasta').write_text(''.join(records))
    with (directory / 'train_terms.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['EntryID', 'term', 'aspect'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows + [rows[0]])


def length_summary(sequences):
    values = np.asarray([len(s) for s in sequences])
    return {key: float(value) for key, value in zip(
        ['min', 'p25', 'median', 'p75', 'max', 'mean'],
        [*np.percentile(values, [0, 25, 50, 75, 100]), values.mean()])}


def run(args):
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        args.data_dir = args.output / 'fixture_inputs'
        make_fixture(args.data_dir)
    if args.data_dir is None:
        raise ValueError('Provide --data-dir or use --smoke-test')
    paths = {filename: args.data_dir / filename for filename in INPUT_FILES}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f'Missing input: {path}. Run download_data.py first.')
    mode = 'SYNTHETIC SMOKE TEST — no real-data metrics' if args.smoke_test else 'Preliminary real-data internal validation'
    print(mode, flush=True)
    nodes, aliases, ancestors, ontology_metadata = load_ontology(paths['go-basic.obo'])
    sequences, fasta_stats, headers = load_fasta(paths['train_sequences.fasta'])
    labels, annotation_ids, annotation_stats, tsv_columns = load_annotations(
        paths['train_terms.tsv'], nodes, aliases, ancestors)
    joined = set(sequences) & annotation_ids
    eligible = sorted(p for p in joined if sequences[p] and labels.get(p))
    if len(eligible) < 2:
        raise ValueError('Too few proteins with nonempty sequences and non-root MF annotations')
    groups = make_groups(eligible, sequences, args.groups)
    train, validation = grouped_split(eligible, groups, args.seed, args.validation_fraction, args.max_proteins)
    assert_split(train, validation, groups, sequences)
    support = Counter(term for protein in train for term in labels[protein])
    terms = sorted(t for t, count in support.items() if count >= args.min_support)
    if not terms:
        raise ValueError('No terms pass the training support threshold; increase pilot size.')
    y_train = target_matrix(train, labels, terms)
    y_validation = target_matrix(validation, labels, terms)
    # Both methods score all validation proteins. Recall excludes only zero-truth
    # rows by definition; AP and precision retain them, avoiding silent filtering.
    columns = ancestor_columns(terms, ancestors)
    thresholds = np.linspace(0, 1, 101)
    split_type = 'supplied-group validation' if args.groups else 'preliminary exact-duplicate-grouped validation'
    selected = set(train + validation)
    missing_sequence_ids = sorted(annotation_ids - sequences.keys())
    missing_annotation_ids = sorted(sequences.keys() - annotation_ids)
    write_csv(args.output / 'unmatched_ids.csv', ['accession', 'reason'],
              [dict(accession=p, reason='annotation_without_sequence') for p in missing_sequence_ids] +
              [dict(accession=p, reason='sequence_without_annotation') for p in missing_annotation_ids])
    write_csv(args.output / 'excluded_proteins.csv', ['accession', 'reason'],
              [dict(accession=p, reason='empty_sequence' if not sequences[p] else 'no_nonroot_mf_labels')
               for p in sorted(joined - set(eligible))])
    write_csv(args.output / 'split.csv', ['accession', 'group_id', 'split'],
              [dict(accession=p, group_id=groups[p], split=split)
               for split, ids in [('train', train), ('validation', validation)] for p in ids])
    validation_support = y_validation.sum(axis=0)
    write_csv(args.output / 'term_support.csv', ['column', 'go_id', 'name', 'train_positive', 'validation_positive'],
              [dict(column=i, go_id=t, name=nodes[t].get('name', ''), train_positive=support[t],
                    validation_positive=int(validation_support[i])) for i, t in enumerate(terms)])
    provenance_path = args.data_dir / 'download_provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {'source': 'user-supplied local files'}
    summary = dict(
        mode=mode, generated_at_utc=datetime.now(timezone.utc).isoformat(),
        command=shlex.join([sys.executable, *sys.argv]),
        inputs={f: dict(path=str(p.resolve()), sha256=sha256(p), bytes=p.stat().st_size) for f, p in paths.items()},
        download_provenance=provenance, ontology=ontology_metadata,
        fasta_header_examples=headers, annotation_columns=tsv_columns,
        **fasta_stats, **annotation_stats,
        joined_proteins=len(joined), mf_labeled_proteins=len(eligible),
        annotations_without_sequences=len(missing_sequence_ids),
        sequences_without_annotations=len(missing_annotation_ids),
        joined_proteins_excluded=len(joined - set(eligible)),
        mf_propagated_distinct_terms_before_support=len({t for p in eligible for t in labels[p]}),
        mf_distinct_terms_in_selected_train_before_support=len(support), retained_terms=len(terms),
        min_training_support=args.min_support,
        pilot_proteins=len(selected), eligible_proteins_outside_pilot=len(eligible) - len(selected),
        sampling_policy='seeded shuffled whole groups; prefix nearest requested protein cap' if args.max_proteins else 'all eligible groups',
        max_proteins_requested=args.max_proteins, train_proteins=len(train), validation_proteins=len(validation),
        train_fraction_actual=len(train) / len(selected), validation_fraction_actual=len(validation) / len(selected),
        split_type=split_type, seed=args.seed, validation_fraction_requested=args.validation_fraction,
        sequence_length_eligible=length_summary([sequences[p] for p in eligible]),
        sequence_length_pilot=length_summary([sequences[p] for p in train + validation]),
        evaluation_proteins=len(validation), evaluation_exclusions=0,
        recall_eligible_proteins=int((y_validation.sum(axis=1) > 0).sum()),
        zero_retained_truth_validation_proteins=int((y_validation.sum(axis=1) == 0).sum()),
        thresholds=thresholds.tolist(), threshold_comparison='score >= threshold; first maximizer reported',
        label_policy='MF is_a ancestors only; exclude GO:0003674, unknown and obsolete terms; resolve alt_id',
        missing_annotation_policy='Zeros are unobserved, not confirmed biological negatives; metrics use the observed-label convention.',
        feature_policy='421 features: raw length, 20 AA frequencies, 400 adjacent AA-pair frequencies; invalid residues excluded from denominators and break pairs; empty counts -> zeros',
        split_assertions=dict(shared_accessions=0, shared_groups=0, shared_exact_sequences=0),
        groups_file=dict(path=str(args.groups.resolve()), sha256=sha256(args.groups)) if args.groups else None,
        hardware=dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                      python=platform.python_version(), threads=1),
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__),
        logistic_settings=dict(solver='liblinear', C=1.0, max_iter=1000, tol=1e-4, random_state=args.seed,
                               scaling='StandardScaler fitted on training rows only'),
    )
    print(f'{len(eligible):,} eligible; pilot {len(train):,} train / {len(validation):,} validation; {len(terms)} labels', flush=True)
    preprocessing_seconds = time.perf_counter() - start
    t0 = time.perf_counter()
    prevalence = y_train.mean(axis=0)
    frequency_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    frequency_scores = propagate_scores(np.tile(prevalence, (len(validation), 1)), columns)
    frequency_inference = time.perf_counter() - t0
    frequency_metrics, frequency_curve = evaluate(y_validation, frequency_scores, thresholds)
    metrics = [dict(method='training_frequency', **frequency_metrics, fit_seconds=frequency_fit,
                    inference_seconds=frequency_inference, batch_size=len(validation))]
    print('Frequency baseline complete; fitting logistic classifiers...', flush=True)
    t0 = time.perf_counter()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(featurize([sequences[p] for p in train]))
    models, convergence_terms = [], []
    for i, term in enumerate(terms):
        if y_train[:, i].min() == y_train[:, i].max():
            models.append(float(prevalence[i]))
        else:
            model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000,
                                       tol=1e-4, random_state=args.seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(x_train, y_train[:, i])
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                convergence_terms.append(term)
            models.append(model)
        if (i + 1) % 50 == 0:
            print(f'  fitted {i + 1}/{len(terms)} terms', flush=True)
    logistic_fit = time.perf_counter() - t0
    del x_train
    t0 = time.perf_counter()
    x_validation = scaler.transform(featurize([sequences[p] for p in validation]))
    logistic_scores = np.empty((len(validation), len(terms)), dtype=np.float64)
    for i, model in enumerate(models):
        logistic_scores[:, i] = model if isinstance(model, float) else model.predict_proba(x_validation)[:, 1]
    propagate_scores(logistic_scores, columns)
    logistic_inference = time.perf_counter() - t0
    logistic_metrics, logistic_curve = evaluate(y_validation, logistic_scores, thresholds)
    metrics.append(dict(method='logistic_regression', **logistic_metrics, fit_seconds=logistic_fit,
                        inference_seconds=logistic_inference, batch_size=len(validation)))
    write_csv(args.output / 'metrics.csv', list(metrics[0]), metrics)
    write_csv(args.output / 'threshold_metrics.csv', ['method', *frequency_curve[0]],
              [dict(method=method, **row) for method, curve in [('training_frequency', frequency_curve),
                                                               ('logistic_regression', logistic_curve)] for row in curve])
    np.savez_compressed(args.output / 'validation_predictions.npz',
                        accessions=np.asarray(validation), go_ids=np.asarray(terms), targets=y_validation,
                        training_frequency=frequency_scores, logistic_regression=logistic_scores)
    example_index = next(i for i in range(len(validation)) if y_validation[i].any())
    example = validation[example_index]
    top = np.argsort(-logistic_scores[example_index], kind='stable')[:10]
    example_rows = [dict(accession=example, go_id=terms[i], name=nodes[terms[i]].get('name', ''),
                         score=float(logistic_scores[example_index, i]),
                         annotated_positive=int(y_validation[example_index, i])) for i in top]
    write_csv(args.output / 'example_predictions.csv', list(example_rows[0]), example_rows)
    summary.update(preprocessing_seconds=preprocessing_seconds,
                   constant_logistic_terms=sum(isinstance(m, float) for m in models),
                   nonconverged_terms=convergence_terms,
                   total_pipeline_seconds=time.perf_counter() - start)
    write_json(args.output / 'data_summary.json', summary)
    report = [f'# CAFA5 check-in: {mode}', '', f'Validation: **{split_type}**.', '',
              '## Reproduce', '', '```sh', summary['command'], '```', '',
              '## Data and method', '',
              f'- Source: {provenance["source"]}.',
              f'- Raw sequence records: {fasta_stats["fasta_records"]:,}; unique sequence accessions: {len(sequences):,}; joined: {len(joined):,}.',
              f'- Eligible non-root MF proteins: {len(eligible):,}; pilot: {len(selected):,}; outside pilot: {len(eligible) - len(selected):,}.',
              f'- Train / validation: {len(train):,} / {len(validation):,} ({summary["validation_fraction_actual"]:.1%} validation); seed {args.seed}.',
              f'- Retained terms: {len(terms)} with at least {args.min_support} training positives after is_a propagation.',
              '- Split checks passed: zero shared accessions, groups, or exact sequences.',
              f'- Evaluation: all {len(validation):,} validation proteins for both methods; {summary["recall_eligible_proteins"]:,} have retained truth for recall.',
              '- Both use the same ordered labels and proteins, with max-score propagation to retained ancestors through the full ontology.',
              '- Logistic regression: 421 sequence features, training-only StandardScaler, liblinear, C=1, max_iter=1000, one CPU thread.', '',
              '## Preliminary internal results', '',
              '| Method | Micro AP | Restricted MF F-max | Threshold | Fit (s) | Inference (s) |',
              '|---|---:|---:|---:|---:|---:|']
    for row in metrics:
        report.append(f'| {row["method"]} | {row["micro_ap"]:.4f} | {row["restricted_mf_fmax"]:.4f} | {row["fmax_threshold"]:.2f} | {row["fit_seconds"]:.4f} | {row["inference_seconds"]:.4f} |')
    difference = logistic_metrics['micro_ap'] - frequency_metrics['micro_ap']
    report += ['', f'Logistic regression micro AP minus frequency micro AP: {difference:+.4f} on this holdout.', '',
               f'Total pipeline time: {summary["total_pipeline_seconds"]:.2f} s. Inference batch: {len(validation):,} proteins.',
               f'Hardware: {summary["hardware"]["platform"]}, {summary["hardware"]["machine"]}; Python {platform.python_version()}; one CPU thread.',
               'Fit time includes training feature extraction/scaling. Inference includes validation feature extraction, scaling, prediction, and hierarchy postprocessing; input file loading and evaluation are excluded. Timings are single-run measurements.', '',
               '## Example output', '', f'Validation accession: `{example}`. Unannotated does not mean biologically absent.', '',
               '| GO term | Name | Logistic score | Annotated positive |', '|---|---|---:|---:|']
    report += [f'| {r["go_id"]} | {r["name"]} | {r["score"]:.4f} | {r["annotated_positive"]} |' for r in example_rows]
    report += ['', '## Limits and next step', '',
               '- Exact duplicates are grouped; close homologs may still cross the split. Similarity-grouped validation remains open.' if not args.groups else '- Group provenance/settings must be documented externally; no claim about a specific similarity threshold is made by this script.',
               '- Scores concern frequent retained MF terms only; these are not official CAFA temporal benchmark scores.',
               '- Zeros denote missing annotations. AP uses the observed-label convention; precision can penalize plausible but unannotated functions.',
               '- F-max sweeps 101 thresholds from 0.00 to 1.00 inclusive (score >= threshold). Its maximizing threshold is descriptive for this holdout, not a tuned deployment threshold.',
               f'- {summary["zero_retained_truth_validation_proteins"]} validation proteins have no retained truth: included in AP and precision; excluded from recall by definition.',
               f'- Nonconverged logistic terms: {len(convergence_terms)} (IDs in data_summary.json).',
               '- Next: similarity-grouped validation, then a BLAST/DIAMOND homology baseline and grouped tuning.', '']
    if args.max_proteins:
        report += ['- This is a seeded whole-group pilot subset, not a full-training-data experiment.', '']
    if args.smoke_test:
        report += ['**These are synthetic smoke-test outputs. Real-data performance remains pending.**', '']
    if 'url' in provenance:
        report += [f'Data source: [{provenance["source"]}]({provenance["url"]}).', '']
    (args.output / 'CHECKIN_RESULTS.md').write_text('\n'.join(report))
    print(f'Finished in {summary["total_pipeline_seconds"]:.2f}s; report: {args.output / "CHECKIN_RESULTS.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/pilot'))
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-support', type=int)
    parser.add_argument('--max-proteins', type=int, help='Optional seeded whole-group pilot cap (approximate)')
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--groups', type=Path, help='Optional CSV: accession,group_id; exact duplicates must agree')
    args = parser.parse_args()
    if args.smoke_test and (args.data_dir or args.groups):
        parser.error('--smoke-test cannot be mixed with real inputs or groups')
    args.min_support = args.min_support if args.min_support is not None else (8 if args.smoke_test else 100)
    if args.min_support < (1 if args.smoke_test else 100):
        parser.error('Support below 100 is allowed only for synthetic smoke tests')
    if not 0 < args.validation_fraction < 1 or (args.max_proteins is not None and args.max_proteins < 2):
        parser.error('Validation fraction must lie in (0,1); pilot cap must be >= 2')
    try:
        with threadpool_limits(limits=1):
            run(args)
    except ValueError as error:
        parser.exit(1, f'Pilot failed: {error}\n')


if __name__ == '__main__':
    main()
