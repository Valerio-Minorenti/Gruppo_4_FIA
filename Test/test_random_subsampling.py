import unittest
import numpy as np
import pandas as pd
from Validation.Random_Subsampling import RandomSubsampling
from kNN_classifier import KNN

class TestRandomSubsampling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.df = pd.DataFrame(
            np.random.rand(100, 5) * 10,
            columns=[f"Feature{i}" for i in range(1, 6)]
        )
        cls.df["Target"] = np.random.choice([0, 1], size=100)
        cls.k_experiments = 5
        cls.test_size = 0.2
        cls.classifier_params = {"k": 3}

    def setUp(self):
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
        x_train, x_test, y_train, y_test = self.subsampling.train_test_split()
        total_samples = self.df.shape[0]
        expected_test_size = int(total_samples * self.test_size)
        expected_train_size = total_samples - expected_test_size

        self.assertEqual(len(x_test), expected_test_size, "La dimensione del test set non è corretta.")
        self.assertEqual(len(x_train), expected_train_size, "La dimensione del training set non è corretta.")
        self.assertEqual(len(y_test), expected_test_size, "La dimensione delle etichette di test non corrisponde.")
        self.assertEqual(len(y_train), expected_train_size, "La dimensione delle etichette di training non corrisponde.")

    def test_predictions_length(self):
        # Usare generate_splits() anziché run_experiments()
        results = self.subsampling.generate_splits()
        for y_test, y_pred in results:
            self.assertEqual(
                len(y_test),
                len(y_pred),
                "Il numero di predizioni non corrisponde al numero di etichette reali."
            )

    def test_number_of_experiments(self):
        # Usare generate_splits() anziché run_experiments()
        results = self.subsampling.generate_splits()
        self.assertEqual(
            len(results),
            self.k_experiments,
            "Il numero di esperimenti eseguiti non corrisponde al valore atteso."
        )

if __name__ == "__main__":
    unittest.main()
