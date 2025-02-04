import unittest
import numpy as np
import pandas as pd
from Validation.Stratified_Cross_Validation import StratifiedCrossValidation  # Assicurati che il file si chiami Stratified_Cross_Validation.py


class TestStratifiedCrossValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Imposta un dataset di test prima di eseguire i test.
        """
        np.random.seed(42)  # Per riproducibilità dei risultati

        # Creiamo un dataset di esempio con 12 campioni e 3 feature
        cls.df = pd.DataFrame({
            "feature1": np.random.randint(1, 10, 12),
            "feature2": np.random.rand(12) * 10,
            "Class": np.random.choice([2, 4], size=12, p=[0.5, 0.5])  # Due classi con distribuzione 50%
        })

        # Parametri per la Stratified Cross-Validation
        cls.K = 3
        cls.k_neighbors = 3

    def setUp(self):
        """
        Eseguito prima di ogni test per creare un'istanza di StratifiedCrossValidation.
        """
        self.cross_validator = StratifiedCrossValidation(
            K=self.K,
            df=self.df,
            class_column="Class",
            k_neighbors=self.k_neighbors
        )

    def test_number_of_folds(self):
        """
        Testa se il dataset è suddiviso in K fold corretti.
        """
        folds = self.cross_validator.split(self.df.copy(), "Class")
        self.assertEqual(len(folds), self.K, "Il numero di fold non corrisponde al valore atteso.")

    def test_predictions_length(self):
        """
        Testa se il numero di predizioni corrisponde al numero di etichette reali nel test set.
        """
        results = self.cross_validator.run_experiments()

        for y_test, y_pred in results:
            self.assertEqual(len(y_test), len(y_pred), "Il numero di predizioni non corrisponde al numero di etichette reali.")

    def test_stratification(self):
        """
        Testa se le classi nei test set rispettano la distribuzione originale.
        """
        folds = self.cross_validator.split(self.df.copy(), "Class")
        original_distribution = self.df["Class"].value_counts(normalize=True)

        for _, test_set in folds:
            test_distribution = test_set["Class"].value_counts(normalize=True)
            for class_label in original_distribution.index:
                self.assertAlmostEqual(
                    original_distribution[class_label], test_distribution.get(class_label, 0),
                    delta=0.2,
                    msg=f"La distribuzione della classe {class_label} nel test set non è coerente con l'originale."
                )

if __name__ == "__main__":
    unittest.main()