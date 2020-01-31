import unittest
import metrics
import pandas as pd

class TestMetrics(unittest.TestCase):
    def setUp(self):
            self.labels = [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
                           1, 0]
            self.scores1 = [0.9, 0.8, 0.7, 0.6, 0.55, 0.51, 0.49, 0.43, 0.42,
                           0.39, 0.33, 0.31, 0.23, 0.22, 0.19, 0.15, 0.12, 0.11,
                           0.04, 0.01]
            self.scores2 = [0.9, 0.8, 0.07, 0.6, 0.55, 0.51, 0.49, 0.43, 0.42,
                           0.39, 0.33, 0.31, 0.23, 0.22, 0.19, 0.15, 0.12, 0.11,
                           0.84, 0.01]        
            
            self.df = pd.DataFrame({'label':self.labels,
                                    'scores1':self.scores1,
                                    'scores2':self.scores2})

    def test_cummulative_gains_array_input(self):
        self.assertAlmostEqual(metrics.cummulative_gains(self.labels,
                                                         self.scores1),
                               0.5875, places=4)
    
    def test_cummulative_gains_df_input_single_score(self):
        self.assertAlmostEqual(metrics.cummulative_gains('label', 'scores1',
                                                         self.df),
                               0.5875, places=4)

    def test_cummulative_gains_df_input_mutiple_socres(self):
        res = metrics.cummulative_gains('label', ['scores1', 'scores2'], self.df)
        self.assertAlmostEqual(res['scores1'], 0.5875, places=4)
        self.assertAlmostEqual(res['scores2'], 0.6675, places=4)

if __name__ == '__main__':
    unittest.main()