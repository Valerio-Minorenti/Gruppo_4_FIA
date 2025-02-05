import unittest
import numpy as np
import pandas as pd
from kNN_classifier import KNN
from Validation.Stratified_Cross_Validation import StratifiedCrossValidation

class TestStratifiedCrossValidation(unittest.TestCase):

    def setUp(self):
        # Creazione di un dataset fittizio
        data = {
            'feature1': np.random.rand(20),
            'feature2': np.random.rand(20),
            'class': [2] * 10 + [4] * 10  # Due classi bilanciate
        }
        self.df = pd.DataFrame(data)
        self.K = 5  # Numero di fold
        self.k_neighbors = 3  # Numero di vicini per KNN
        self.validator = StratifiedCrossValidation(self.K, self.df, 'class', self.k_neighbors)

    def test_split_folds_count(self):
        """Testa se la divisione in fold avviene correttamente."""
        folds = self.validator.split(self.df, 'class')
        self.assertEqual(len(folds), self.K, "Il numero di fold generati non è corretto.")

    def test_class_proportions_preserved(self):
        """Verifica che le proporzioni delle classi siano preservate nei fold."""
        folds = self.validator.split(self.df, 'class')
        for train_set, test_set in folds:
            train_class_counts = train_set['class'].value_counts(normalize=True)
            test_class_counts = test_set['class'].value_counts(normalize=True)
            self.assertAlmostEqual(train_class_counts[2], 0.5, delta=0.1,
                                   msg="La proporzione della classe 2 nel training set non è mantenuta.")
            self.assertAlmostEqual(train_class_counts[4], 0.5, delta=0.1,
                                   msg="La proporzione della classe 4 nel training set non è mantenuta.")
            self.assertAlmostEqual(test_class_counts[2], 0.5, delta=0.2,
                                   msg="La proporzione della classe 2 nel test set non è mantenuta.")
            self.assertAlmostEqual(test_class_counts[4], 0.5, delta=0.2,
                                   msg="La proporzione della classe 4 nel test set non è mantenuta.")

    def test_run_experiments_output(self):
        """Testa se il metodo run_experiments restituisce l'output atteso."""
        results = self.validator.run_experiments()
        self.assertEqual(len(results), self.K, "Il numero di risultati non corrisponde ai fold previsti.")
        for y_test, predictions, predicted_proba in results:
            self.assertEqual(len(y_test), len(predictions), "Le dimensioni di y_test e predictions non coincidono.")
            self.assertEqual(len(y_test), len(predicted_proba),
                             "Le dimensioni di y_test e predicted_proba non coincidono.")


if __name__ == '__main__':
        unittest.main()