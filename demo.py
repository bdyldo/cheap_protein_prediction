"""Display saved pilot results for the TA without rerunning training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('outputs/real_pilot'))
    parser.add_argument('--protein', help='Optional validation accession to inspect')
    args = parser.parse_args()
    try:
        summary = json.loads((args.results / 'data_summary.json').read_text())
        with (args.results / 'metrics.csv').open() as handle:
            metrics = list(csv.DictReader(handle))
        with (args.results / 'term_support.csv').open() as handle:
            names = {r['go_id']: r['name'] for r in csv.DictReader(handle)}
    except FileNotFoundError:
        parser.exit(1, 'Saved results are missing. Run pilot.py first.\n')
    print(summary['mode'])
    print(summary['split_type'])
    print(f"{summary['train_proteins']:,} training / {summary['validation_proteins']:,} validation proteins; "
          f"{summary['retained_terms']} retained MF terms")
    print('Shared accessions / groups / exact sequences: 0 / 0 / 0')
    print(f"\n{'Method':24} {'Micro AP':>9} {'MF F-max':>9} {'Fit (s)':>10} {'Inference (s)':>14}")
    for row in metrics:
        print(f"{row['method']:24} {float(row['micro_ap']):9.4f} "
              f"{float(row['restricted_mf_fmax']):9.4f} {float(row['fit_seconds']):10.3f} "
              f"{float(row['inference_seconds']):14.4f}")
    with np.load(args.results / 'validation_predictions.npz', allow_pickle=False) as saved:
        ids, terms, targets = saved['accessions'], saved['go_ids'], saved['targets']
        if args.protein:
            matches = np.flatnonzero(ids == args.protein)
            if not len(matches):
                parser.error('That accession is not in the validation batch')
            index = int(matches[0])
        else:
            index = int(np.flatnonzero(targets.any(axis=1))[0])
        scores = saved['logistic_regression'][index]
        print(f'\nExample validation protein: {ids[index]}')
        for column in np.argsort(-scores, kind='stable')[:10]:
            label = 'annotated' if targets[index, column] else 'unannotated'
            print(f'{terms[column]}  {scores[column]:.4f}  {label:11}  {names[terms[column]]}')
    print('\nPreliminary restricted-MF holdout results. Unannotated does not mean absent.')
    print('Next: similarity-grouped validation and a homology baseline.')


if __name__ == '__main__':
    main()
