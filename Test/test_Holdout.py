import unittest
import numpy as np
from Validation.Holdout import Holdouts


class TestHoldouts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Configura un dataset di test prima di eseguire i test.
        """
        np.random.seed(42)  # Per risultati riproducibili

        # Dataset di esempio con 100 campioni
        cls.data = np.random.rand(100, 5) * 10  # Valori casuali tra 0 e 10
        cls.labels = np.random.choice([0, 1], size=100)  # Etichette binarie (0 o 1)

        cls.test_ratio = 0.2  # 20% per il testing
        cls.k = 3  # Numero di vicini per kNN

    def setUp(self):
        """
        Eseguito prima di ogni test.
        """
        self.holdout = Holdouts(self.test_ratio, self.data, self.labels)

    def test_split_correctness(self):
        """
        Testa se il dataset è diviso correttamente in training e test set.
        """
        num_samples = len(self.data)
        num_test_samples = int(num_samples * self.test_ratio)
        num_train_samples = num_samples - num_test_samples

        # Esegui la suddivisione
        results = self.holdout.generate_splits(self.k)
        y_test, _ = results[0]  #  etichette reali

        self.assertEqual(len(y_test), num_test_samples, "Il numero di campioni nel test set è errato")

    def test_knn_predictions(self):
        """
        Testa se il kNN genera previsioni della stessa lunghezza delle etichette reali.
        """
        results = self.holdout.generate_splits(self.k)
        y_test, y_pred = results[0]  # Prendiamo etichette reali e predette

        self.assertEqual(len(y_test), len(y_pred), "Le previsioni del kNN non corrispondono al numero di test sample")

    def test_knn_values_are_binary(self):
        """
        Testa se le previsioni del kNN sono valori binari (0 o 1).
        """
        results = self.holdout.generate_splits(self.k)
        _, y_pred = results[0]

        unique_values = set(y_pred)
        self.assertTrue(unique_values.issubset({0, 1}), "Il kNN dovrebbe restituire solo 0 o 1")

    def test_invalid_test_ratio(self):
        """
        Verifica che venga sollevato un errore se il test_ratio è fuori dai limiti validi.
        """
        with self.assertRaises(ValueError):
            Holdouts(1.5, self.data, self.labels)  # test_ratio > 1 non è valido

    def test_empty_training_set(self):
        """
        Verifica che venga sollevato un errore se il training set è vuoto.
        """
        with self.assertRaises(ValueError):
            Holdouts(1, self.data, self.labels).generate_splits(self.k)  # Troppo poco training set

if __name__ == "__main__":
    unittest.main()