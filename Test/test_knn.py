import unittest
import numpy as np
from kNN_classifier import KNN  # Assicurati che il file si chiami KNN_classifier.py

class TestKNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Crea un dataset di esempio per i test.
        """
        cls.x_train = np.array([
            [1, 2], [2, 3], [3, 3], [5, 5], [8, 8]
        ])
        cls.y_train = np.array([0, 1, 1, 0, 1])
        cls.knn = KNN(k=3)
        cls.knn.fit(cls.x_train, cls.y_train)

    def test_euclidean_distance(self):
        """
        Testa se euclidean_distance() calcola correttamente la distanza tra due punti.
        """
        point1 = np.array([1, 1])
        point2 = np.array([4, 5])
        expected_distance = np.linalg.norm(point1 - point2)
        calculated_distance = self.knn.euclidean_distance(point1, point2)

        self.assertAlmostEqual(calculated_distance, expected_distance, places=4, 
                               msg="La distanza euclidea calcolata non è corretta.")

    def test_nearest_neighbors(self):
        """
        Testa se nearest_neighbors() restituisce i k vicini più vicini.
        """
        test_sample = np.array([4, 4])
        neighbors = self.knn.nearest_neighbors(test_sample)

        self.assertEqual(len(neighbors), self.knn.k, "Il numero di vicini trovati non è corretto.")
        self.assertTrue(all(n in range(len(self.x_train)) for n in neighbors),
                        "Gli indici restituiti non appartengono al dataset di training.")

    def test_predict(self):
        """
        Testa se predict() restituisce un'etichetta valida per un campione di test.
        """
        test_samples = np.array([[3, 4], [6, 6]])
        predictions = self.knn.predict(test_samples)

        self.assertEqual(len(predictions), len(test_samples), "Il numero di predizioni non è corretto.")
        self.assertTrue(all(p in self.y_train for p in predictions),
                        "Le predizioni contengono etichette non valide.")

if __name__ == "__main__":
    unittest.main()