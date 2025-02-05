import unittest
import numpy as np
import pandas as pd
from kNN_classifier import KNN  # Importa la classe KNN personalizzata
from Validation.Validation_Strategy import ValidationStrategy  # Importa la classe astratta
from Validation.Holdout import Holdouts  # Importa la classe Holdouts


class TestHoldouts(unittest.TestCase):

    def setUp(self):
        """Imposta i dati di esempio per i test."""
        np.random.seed(42)  # Imposta un seed fisso per la riproducibilità

        # Esempio di dati
        self.sample_data = {
            'feature1': [5, 2, 7, 1, 9, 4],
            'feature2': [0.2, 0.5, 0.1, 0.9, 1.2, 0.3],
            'Class': [2, 2, 4, 2, 4, 4]
        }
        self.df_example = pd.DataFrame(self.sample_data)
        self.holdout = Holdouts(
            test_ratio=0.3,
            data=self.df_example[['feature1', 'feature2']],
            labels=self.df_example['Class']
        )

    def test_train_test_split(self):
        """Verifica che lo split dei dati avvenga correttamente."""
        data, labels = self.holdout.features, self.holdout.labels
        num_samples = len(data)
        num_test_samples = int(num_samples * self.holdout.test_ratio)
        shuffled_indices = np.random.permutation(num_samples)
        test_indices = shuffled_indices[:num_test_samples]
        train_indices = shuffled_indices[num_test_samples:]

        x_train = data.iloc[train_indices].to_numpy()
        x_test = data.iloc[test_indices].to_numpy()
        y_train = labels.iloc[train_indices].to_numpy()
        y_test = labels.iloc[test_indices].to_numpy()

        # Calcolo delle dimensioni attese
        expected_train_size = num_samples - num_test_samples
        expected_test_size = num_test_samples

        # Asserzioni
        self.assertEqual(len(x_train), expected_train_size)
        self.assertEqual(len(x_test), expected_test_size)
        self.assertEqual(len(y_train), expected_train_size)
        self.assertEqual(len(y_test), expected_test_size)

    def test_generate_splits(self):
        """Verifica il funzionamento del metodo generate_splits()."""
        results = self.holdout.generate_splits(k=3)
        self.assertEqual(len(results), 1)

        y_test, predictions, predicted_proba_continuous = results[0]

        # Controlla che le dimensioni delle previsioni e del test set siano coerenti
        self.assertEqual(len(y_test), int(len(self.holdout.features) * self.holdout.test_ratio))
        self.assertEqual(len(predictions), len(y_test))
        self.assertEqual(len(predicted_proba_continuous), len(y_test))

    def test_predict_proba(self):
        """Verifica che le probabilità previste siano valide."""
        data, labels = self.holdout.features, self.holdout.labels
        num_samples = len(data)
        num_test_samples = int(num_samples * self.holdout.test_ratio)
        shuffled_indices = np.random.permutation(num_samples)
        test_indices = shuffled_indices[:num_test_samples]
        train_indices = shuffled_indices[num_test_samples:]

        x_train = data.iloc[train_indices].to_numpy()
        x_test = data.iloc[test_indices].to_numpy()
        y_train = labels.iloc[train_indices].to_numpy()
        y_test = labels.iloc[test_indices].to_numpy()

        classifier = KNN(k=3)
        classifier.fit(x_train, y_train)
        predicted_proba = classifier.predict_proba(x_test)

        # Calcola le probabilità continue per la classe positiva
        positive_class = 4
        predicted_proba_continuous = [proba.get(positive_class, 0.0) for proba in predicted_proba]

        # Controlli sulle probabilità
        self.assertEqual(len(predicted_proba_continuous), len(y_test))
        for proba in predicted_proba_continuous:
            self.assertTrue(0.0 <= proba <= 1.0)


if __name__ == '__main__':
    unittest.main()
