import unittest
import numpy as np
import pandas as pd
from kNN_classifier import KNN
from Validation.Random_Subsampling import RandomSubsampling

class TestRandomSubsampling(unittest.TestCase):

    def setUp(self):
        # Esempio di dati per i test
        self.sample_data = {
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [10, 9, 8, 7, 6, 5],
            'Class':    [2,  4, 4, 2, 2, 4]
        }
        self.df_example = pd.DataFrame(self.sample_data)
        self.random_sub = RandomSubsampling(
            df=self.df_example,
            x=self.df_example[['feature1', 'feature2']],
            y='Class',
            k_experiments=3,
            classifier_class=KNN,
            classifier_params={'k': 3},
            test_size=0.3
        )

    def test_train_test_split(self):
        x_train, x_test, y_train, y_test = self.random_sub.train_test_split()
        self.assertEqual(len(x_train), 4)
        self.assertEqual(len(x_test), 2)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 2)

    def test_generate_splits(self):
        results = self.random_sub.generate_splits(k=1)
        self.assertEqual(len(results), 1)
        y_test, predictions, predicted_proba_continuous = results[0]
        self.assertEqual(len(y_test), 2)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(predicted_proba_continuous), 2)

    def test_predict_proba(self):
        x_train, x_test, y_train, y_test = self.random_sub.train_test_split()
        classifier = KNN(k=3)
        classifier.fit(x_train, y_train)
        predicted_proba = classifier.predict_proba(x_test)
        positive_class = 4
        predicted_proba_continuous = [proba.get(positive_class, 0.0) for proba in predicted_proba]
        self.assertEqual(len(predicted_proba_continuous), 2)
        for proba in predicted_proba_continuous:
            self.assertTrue(0.0 <= proba <= 1.0)

if __name__ == '__main__':
    unittest.main()