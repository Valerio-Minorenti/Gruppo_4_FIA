import pandas as pd
import numpy as np
from kNN_classifier import KNN
from Validation.Validation_Strategy import ValidationStrategy  # Importa la classe astratta

class Holdouts(ValidationStrategy):
    def __init__(self, test_ratio, data):
        """
        Inizializza i parametri per la strategia di validazione Holdout.

        Args:
            test_ratio (float): Percentuale per il testing (0 < test_ratio < 1).
            data (np.ndarray o pd.DataFrame): Dataset delle feature.
        """
        if not (0 < test_ratio < 1):
            raise ValueError("test_ratio deve essere un valore tra 0 e 1 (es. 0.2 per 20%).")

        self.test_ratio = test_ratio
        # Convertiamo in DataFrame per coerenza interna
        self.features = data.drop(columns=['classtype_v1'])
        self.labels = data['classtype_v1']  # Utilizza la colonna 'classtype_v1' come etichetta

    def generate_splits(self, k=None):
        """
        Implementazione del metodo astratto di ValidationStrategy.
        Divide il dataset in training e testing, allena il kNN e restituisce le previsioni.

        Args:
            k (int, opzionale): Numero di vicini per il kNN.
                                Se non fornito, lancia un errore o imposta un default.

        Returns:
            list[tuple[list[int], list[int]]]: Lista con una singola tupla (y_veri, y_predetti).
        """
        if k is None:
            # Puoi decidere un default o lanciare un errore
            raise ValueError("Parametro 'k' non specificato. Devi fornire il numero di vicini per il kNN.")

        # Ricaviamo data e labels
        data, labels = self.features, self.labels
        num_samples = len(data)
        num_test_samples = int(num_samples * self.test_ratio)

        if num_test_samples == num_samples:
            raise ValueError("Il training set è vuoto. Riduci il valore di test_ratio.")

        # Creiamo indici mescolati
        shuffled_indices = np.random.permutation(num_samples)
        test_indices = shuffled_indices[:num_test_samples]
        train_indices = shuffled_indices[num_test_samples:]

        # Suddivisione in train/test
        x_train = data.iloc[train_indices].to_numpy()
        x_test  = data.iloc[test_indices].to_numpy()
        y_train = labels.iloc[train_indices].to_numpy()
        y_test  = labels.iloc[test_indices].to_numpy()

        # Inizializza il kNN con k vicini
        knn_model = KNN(k)
        knn_model.fit(x_train, y_train)

        # Calcola le probabilità previste sul test set
        predicted_proba = knn_model.predict_proba(x_test)

        # Converti le probabilità della classe positiva in un array continuo
        positive_class = 4
        predicted_proba_continuous = [proba.get(positive_class, 0.0) for proba in predicted_proba]
        predicted_proba_continuous = [float(predprob) for predprob in predicted_proba_continuous]

        # Predice sul test set
        predictions = knn_model.predict(x_test)
        predictions = [int(pred) for pred in predictions]
        y_test = [int(ytest) for ytest in y_test]

        # Restituisce una lista con una singola tupla (etichette reali, etichette previste)
        results = [(y_test, predictions, predicted_proba_continuous)]
        return results
