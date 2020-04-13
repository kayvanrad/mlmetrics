import unittest
import metrics

class TestMetrics(unittest.TestCase):

    def test_cum_gains_typical_input(self):
        labels = [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
        scores = [0.04, 0.01, 0.9, 0.8, 0.7, 0.6, 0.55, 0.51, 0.49, 0.43, 0.42,
                  0.39, 0.33, 0.31, 0.23, 0.22, 0.19, 0.15, 0.12, 0.11]      
        _,_,auc = metrics.cum_gains(labels, scores)
        self.assertAlmostEqual(auc, 0.5875, places=4)

    def test_cum_gains_empty_inputs(self):
        frac_samples, response_rate, auc = metrics.cum_gains([],[])
        self.assertEqual(frac_samples.size, 0)
        self.assertEqual(response_rate.size, 0)
        self.assertEqual(auc, 0.0)

    def test_cum_gains_unequal_length_inputs(self):
        self.assertRaises(ValueError, metrics.cum_gains, [1], [2,3])



    


if __name__ == '__main__':
    unittest.main()
