"""Checks for metric definitions, hierarchy consistency, and split leakage."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

import pilot


class ScientificChecks(unittest.TestCase):
    def test_accession_formats(self):
        self.assertEqual(pilot.accession('sp|P12345|ABC_HUMAN extra'), 'P12345')
        self.assertEqual(pilot.accession('P12345 sp|P12345|ABC_HUMAN'), 'P12345')

    def test_nonstandard_residues_do_not_bridge_pairs(self):
        features = pilot.featurize(['AXC', 'XX', 'A', 'AC'])
        self.assertEqual(features.shape, (4, 421))
        np.testing.assert_allclose(features[:, 0], [3, 2, 1, 2])
        self.assertEqual(features[0, 1 + pilot.AA_INDEX['A']], 0.5)
        self.assertEqual(features[0, 1 + pilot.AA_INDEX['C']], 0.5)
        np.testing.assert_array_equal(features[:3, 21:], 0)
        self.assertEqual(features[3, 21 + pilot.AA_INDEX['A'] * 20 + pilot.AA_INDEX['C']], 1)
        self.assertTrue(np.isfinite(features).all())

    def test_split_keeps_duplicates_and_is_reproducible(self):
        sequences = {f'P{i}': f'AC{i}' for i in range(20)}
        sequences['duplicate'] = sequences['P0']
        groups = pilot.make_groups(list(sequences), sequences)
        split = pilot.grouped_split(list(sequences), groups, seed=42)
        self.assertEqual(split, pilot.grouped_split(list(reversed(sequences)), groups, seed=42))
        pilot.assert_split(*split, groups, sequences)
        self.assertEqual('P0' in split[0], 'duplicate' in split[0])

    def test_large_group_and_subset_are_never_broken(self):
        sequences = {f'P{i}': 'AC' if i < 30 else f'AC{i}' for i in range(50)}
        groups = pilot.make_groups(list(sequences), sequences)
        train, validation = pilot.grouped_split(list(sequences), groups, seed=42, max_proteins=15)
        pilot.assert_split(train, validation, groups, sequences)
        selected = set(train + validation)
        self.assertIn(len(selected & {f'P{i}' for i in range(30)}), [0, 30])

    def test_fmax_uses_protein_averages_and_unlabeled_rows(self):
        target = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.uint8)
        scores = np.array([[.9, .8], [.7, .1], [.6, .1]])
        metrics, curve = pilot.evaluate(target, scores, [.5])
        self.assertAlmostEqual(metrics['micro_ap'], (1 + 2 / 3 + .5) / 3)
        self.assertAlmostEqual(curve[0]['precision'], .5)
        self.assertAlmostEqual(curve[0]['recall'], .75)
        self.assertAlmostEqual(metrics['restricted_mf_fmax'], .6)

    def test_empty_predictions_have_zero_precision(self):
        metrics, curve = pilot.evaluate(np.array([[1, 0]], dtype=np.uint8), np.array([[.2, .1]]), [.9])
        self.assertEqual(metrics['restricted_mf_fmax'], 0)
        self.assertEqual(curve[0]['predicted_proteins'], 0)

    def test_annotation_dedup_and_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot.make_fixture(root)
            nodes, aliases, ancestors, _ = pilot.load_ontology(root / 'go-basic.obo')
            labels, _, stats, _ = pilot.load_annotations(root / 'train_terms.tsv', nodes, aliases, ancestors)
            self.assertEqual(stats['duplicate_protein_term_rows'], 1)
            self.assertIn('GO:9000001', labels['SYNTHETIC_000'])
            self.assertNotIn(pilot.MF_ROOT, labels['SYNTHETIC_000'])

    def test_scores_reach_ancestors_through_unretained_intermediates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot.make_fixture(root)
            with (root / 'go-basic.obo').open('a') as handle:
                handle.write('\n[Term]\nid: GO:9000004\nname: deep child\nnamespace: molecular_function\nis_a: GO:9000002\n')
            _, _, ancestors, _ = pilot.load_ontology(root / 'go-basic.obo')
            terms = ['GO:9000001', 'GO:9000004']  # GO:9000002 is deliberately absent.
            scores = np.array([[.1, .8]])
            pilot.propagate_scores(scores, pilot.ancestor_columns(terms, ancestors))
            np.testing.assert_allclose(scores, [[.8, .8]])

    def test_supplied_groups_cannot_separate_duplicate_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'groups.csv'
            path.write_text('accession,group_id\nA,1\nB,2\n')
            with self.assertRaisesRegex(ValueError, 'Exact duplicate'):
                pilot.make_groups(['A', 'B'], {'A': 'AC', 'B': 'AC'}, path)


if __name__ == '__main__':
    unittest.main()
