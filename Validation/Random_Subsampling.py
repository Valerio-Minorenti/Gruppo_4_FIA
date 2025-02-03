import numpy as np
import pandas as pd
from kNN_classifier import KNN

class RandomSubsampling:
    def __init__(self, df, x, y, k_experiments, classifier_class, classifier_params, test_size=0.2):
        """
        Inizializza la classe RandomSubsampling con i parametri specificati.

        :param df: DataFrame contenente i dati.
        :param x: Features del dataset.
        :param y: Colonna target (etichette).
        :param k_experiments: Numero di esperimenti da eseguire.
        :param classifier_class: Classe del classificatore da utilizzare.
        :param classifier_params: Dizionario contenente i parametri per inizializzare il classificatore.
        :param test_size: Percentuale di dati da utilizzare per il test.
        """
        self.df = pd.DataFrame(df)
        self.x = x.values if isinstance(x, pd.DataFrame) else np.array(x)
        self.y = df[y].values if isinstance(y, str) else np.array(y)
        self.k_experiments = k_experiments
        self.classifier_class = classifier_class
        self.classifier_params = classifier_params
        self.test_size = test_size

    def train_test_split(self):
        """
        Divide i dati in set di addestramento e di test.

        :return: Tuple contenente i set di addestramento e di test (x_train, x_test, y_train, y_test).
        """
        indices = np.arange(self.x.shape[0])
        np.random.shuffle(indices)
        split_index = int(self.x.shape[0] * (1 - self.test_size))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        return self.x[train_indices], self.x[test_indices], self.y[train_indices], self.y[test_indices]

    def run_experiments(self):
        """
        Esegue il random subsampling per il numero specificato di esperimenti.

        :return: Lista di tuple contenenti le etichette reali e le predizioni per ciascun esperimento.
        """
        results = []
        for _ in range(self.k_experiments):
            x_train, x_test, y_train, y_test = self.train_test_split()
            classifier = self.classifier_class(**self.classifier_params)
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)
            predictions = [int(pred) for pred in predictions]
            results.append((y_test.tolist(), predictions))
        return results