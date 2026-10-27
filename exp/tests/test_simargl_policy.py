import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
import prepare_real as real
from audit_simargl import START_MS, END_MS, HOST, masks


class SimarglPolicyTests(unittest.TestCase):
    def test_bounded_policy_preserves_labels_and_rejects_host_wide_exclusion(self):
        frame = pd.DataFrame({
            'FLOW_START_MILLISECONDS': [START_MS - 1, START_MS, END_MS - 1, END_MS, START_MS, START_MS],
            'IPV4_SRC_ADDR': [HOST, HOST, HOST, HOST, 'other', HOST],
            'LABEL': ['Normal flow'] * 5 + ['FIN Scan']})
        old, bounded = masks(frame)
        self.assertEqual(bounded.tolist(), [False, True, True, False, False, False])
        self.assertTrue(old[0] and old[3])
        real.SIM_KEEP_SCAN[0] = False
        self.assertEqual(real.sim_fam(frame).tolist(), ['BENIGN', 'EXCLUDED', 'EXCLUDED', 'BENIGN', 'BENIGN', 'FIN Scan'])
        real.SIM_KEEP_SCAN[0] = True
        self.assertEqual(real.sim_fam(frame).tolist(), ['BENIGN'] * 5 + ['FIN Scan'])
        real.SIM_KEEP_SCAN[0] = False


if __name__ == '__main__':
    unittest.main()
