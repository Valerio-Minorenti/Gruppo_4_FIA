import pandas as pd
import numpy as np
from kNN_classifier import KNN  # Importa la classe KNN personalizzata

# Classe Holdout
class Holdouts:
    def __init__(self, test_ratio, data, labels):
        """
        Imposta la proporzione tra training e testing.

        Args:
            test_ratio (float): Percentuale per il testing.
            data (np.ndarray): Dataset delle features.
            labels (np.ndarray): Dataset delle etichette.
        """
        if not (0 < test_ratio < 1):
            raise ValueError("test_ratio deve essere un valore tra 0 e 1.")
        self.test_ratio = test_ratio
        self.features = pd.DataFrame(data)
        self.labels = pd.Series(labels)

    def generate_splits(self, k):
        """
        Divide il dataset in training e testing, allena il kNN e restituisce le previsioni.

        Args:
            k (int): Numero di vicini per il kNN.

        Returns:
            list[tuple[list[int], list[int]]]: Lista con singola tupla (y_veri, y_predetti).
        """
        data, labels = self.features, self.labels

        num_samples = len(data)
        num_test_samples = int(num_samples * self.test_ratio)

        if num_test_samples == num_samples:
            raise ValueError("Il training set è vuoto. Riduci il valore di test_ratio.")

        # Gli indici sono mescolati
        shuffled_indices = np.random.permutation(num_samples)
        test_indices = shuffled_indices[:num_test_samples]
        train_indices = shuffled_indices[num_test_samples:]

        # Divide i dati
        x_train, x_test = data.iloc[train_indices].to_numpy(), data.iloc[test_indices].to_numpy()
        y_train, y_test = labels.iloc[train_indices].to_numpy(), labels.iloc[test_indices].to_numpy()

        # Inizializza il kNN
        knn_model = KNN(k)
        knn_model.fit(x_train, y_train)

        # Predice
        predictions = knn_model.predict(x_test)

        # Lista tuple (etichette reali, etichette previste)
        results = [(y_test.tolist(), predictions)]

        return results
