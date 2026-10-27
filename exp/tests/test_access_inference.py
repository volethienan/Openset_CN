import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import stats, holm


class InferenceTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame([dict(dataset=f'D{d}', seed=s, method=m, F1=v)
                             for d in range(7) for s in range(5)
                             for m, v in [('ernp', .8 + .001 * s), ('base', .6 + .001 * s)]])

    def test_seed_replication_does_not_inflate_n_or_change_p(self):
        df = self.data(); extra = df.assign(seed=df.seed + 5)
        first, _, meta = stats(df)
        again, _, meta2 = stats(pd.concat([df, extra]))
        self.assertEqual(meta['n_blocks'], 7); self.assertEqual(meta2['n_blocks'], 7)
        self.assertEqual(first.loc['base', 'p_one_sided'], again.loc['base', 'p_one_sided'])
        self.assertAlmostEqual(first.loc['base', 'p_one_sided'], 1 / 128)

    def test_pair_common_seeds_before_averaging(self):
        df = self.data()
        df.loc[(df.method == 'ernp') & (df.seed == 4), 'F1'] = 100.
        df = df[~((df.method == 'base') & (df.seed == 4))]
        result, _, _ = stats(df)
        self.assertAlmostEqual(result.loc['base', 'mean_diff'], .2)

    def test_scenarios_are_one_source_dataset(self):
        df = self.data(); df['dataset'] = ['CIDDS_ctx_inj' + str(i % 4) for i in range(len(df))]
        # Build explicit unique scenario/seed/method rows.
        df = pd.DataFrame([dict(dataset=f'CIDDS_ctx_inj{i}', seed=s, method=m, F1=v)
                           for i in range(4) for s in range(5) for m, v in [('ernp', .8), ('base', .6)]])
        result, _, meta = stats(df)
        self.assertEqual(meta['n_blocks'], 1)
        self.assertTrue(np.isnan(result.loc['base', 'p_one_sided']))

    def test_duplicate_rows_fail_instead_of_silent_average(self):
        df = self.data()
        with self.assertRaises(ValueError):
            stats(pd.concat([df, df.iloc[:1]]))

    def test_holm_and_unavailable_tests(self):
        np.testing.assert_allclose(holm([.01, .04, .03, np.nan]), [.03, .06, .06, np.nan], equal_nan=True)


if __name__ == '__main__':
    unittest.main()
