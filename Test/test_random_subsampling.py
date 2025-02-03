import unittest
import numpy as np
import pandas as pd
from Validation.Random_Subsampling import RandomSubsampling  # Assicurati che il file si chiami Random_Subsampling.py
from kNN_classifier import KNN  # Importa il tuo classificatore KNN

class TestRandomSubsampling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Imposta un dataset di test prima di eseguire i test.
        """
        np.random.seed(42)  # Per riproducibilità dei risultati

        # Creiamo un dataset di esempio con 100 campioni e 5 feature
        cls.df = pd.DataFrame(np.random.rand(100, 5) * 10, columns=[f"Feature{i}" for i in range(1, 6)])
        cls.df["Target"] = np.random.choice([0, 1], size=100)  # Etichette binarie

        # Parametri per il Random Subsampling
        cls.k_experiments = 5
        cls.test_size = 0.2
        cls.classifier_params = {"k": 3}

    def setUp(self):
        """
        Eseguito prima di ogni test per creare un'istanza di RandomSubsampling.
        """
        self.subsampling = RandomSubsampling(
            df=self.df,
            x=self.df.drop(columns=["Target"]),
            y="Target",
            k_experiments=self.k_experiments,
            classifier_class=KNN,
            classifier_params=self.classifier_params,
            test_size=self.test_size
        )

    def test_train_test_split(self):
        """
        Testa se la divisione train-test rispetta la percentuale test_size.
        """
        x_train, x_test, y_train, y_test = self.subsampling.train_test_split()

        total_samples = self.df.shape[0]
        expected_test_size = int(total_samples * self.test_size)
        expected_train_size = total_samples - expected_test_size

        self.assertEqual(len(x_test), expected_test_size, "La dimensione del test set non è corretta.")
        self.assertEqual(len(x_train), expected_train_size, "La dimensione del training set non è corretta.")

    def test_predictions_length(self):
        """
        Testa se il numero di predizioni corrisponde al numero di etichette reali nel test set.
        """
        results = self.subsampling.run_experiments()

        for y_test, y_pred in results:
            self.assertEqual(len(y_test), len(y_pred), "Il numero di predizioni non corrisponde al numero di etichette reali.")

    def test_number_of_experiments(self):
        """
        Testa se il numero di esperimenti generati è corretto.
        """
        results = self.subsampling.run_experiments()
        self.assertEqual(len(results), self.k_experiments, "Il numero di esperimenti eseguiti non corrisponde al valore atteso.")

if __name__ == "__main__":
    unittest.main()