import numpy as np
import pandas as pd
from kNN_classifier import KNN
from Validation.Validation_Strategy import ValidationStrategy

class RandomSubsampling(ValidationStrategy):
    def __init__(self, df, x, y, k_experiments, classifier_class, classifier_params, test_size=0.2):
        """
        Inizializza la classe RandomSubsampling con i parametri specificati.

        :param df: DataFrame contenente i dati.
        :param x: Features del dataset (può essere DataFrame o array).
        :param y: Colonna target (stringa con il nome nel df, oppure array di etichette).
        :param k_experiments: Numero di esperimenti da eseguire.
        :param classifier_class: Classe del classificatore da utilizzare (es. KNN).
        :param classifier_params: Dizionario dei parametri per inizializzare il classificatore.
        :param test_size: Percentuale di dati da utilizzare per il test (default 0.2).
        """
        # Convertiamo df in DataFrame per sicurezza
        self.df = pd.DataFrame(df)

        # Se x è un DataFrame, estraiamo i valori, altrimenti assicuriamoci sia un np.array
        if isinstance(x, pd.DataFrame):
            self.x = x.values
        else:
            self.x = np.array(x)

        # Se y è una stringa, prendiamo df[y].values; se è già array/list, usiamo np.array(y)
        if isinstance(y, str):
            self.y = self.df[y].values
        else:
            self.y = np.array(y)

        self.k_experiments = k_experiments
        self.classifier_class = classifier_class
        self.classifier_params = classifier_params
        self.test_size = test_size

    def train_test_split(self):
        """
        Suddivide i dati in training e test in modo casuale.

        :return: (x_train, x_test, y_train, y_test)
        """
        indices = np.arange(self.x.shape[0])
        np.random.shuffle(indices)

        split_index = int(self.x.shape[0] * (1 - self.test_size))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        x_train = self.x[train_indices]
        x_test = self.x[test_indices]
        y_train = self.y[train_indices]
        y_test = self.y[test_indices]

        return x_train, x_test, y_train, y_test

    def generate_splits(self, k=None):
        """
        Implementazione del metodo astratto di ValidationStrategy.
        Esegue il random subsampling per 'k' esperimenti (se k non è fornito,
        usa self.k_experiments) e restituisce la lista di tuple (y_test, predictions).

        :param k: Numero di esperimenti da eseguire (opzionale).
        :return: Lista di tuple (y_test, y_pred) per ciascun esperimento.
        """
        if k is None:
            k = self.k_experiments

        results = []
        for _ in range(k):
            x_train, x_test, y_train, y_test = self.train_test_split()

            # Inizializziamo il classificatore (ad es. KNN)
            classifier = self.classifier_class(**self.classifier_params)
            classifier.fit(x_train, y_train)

            predictions = classifier.predict(x_test)
            # Convertiamo a intero se il classificatore restituisce float
            predictions = [int(pred) for pred in predictions]

            # Aggiungiamo la tupla (etichette reali, etichette predette)
            results.append((y_test.tolist(), predictions))

        return results

# (Test opzionale)
if __name__ == '__main__':
    # Esempio di utilizzo
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [10, 9, 8, 7, 6, 5],
        'Class':    [2,  4, 4, 2, 2, 4]
    }
    df_example = pd.DataFrame(sample_data)

    random_sub = RandomSubsampling(
        df=df_example,
        x=df_example[['feature1', 'feature2']],
        y='Class',
        k_experiments=3,
        classifier_class=KNN,
        classifier_params={'k': 3},
        test_size=0.3
    )

    results = random_sub.generate_splits()  # di default k=3
    for i, (y_test, preds) in enumerate(results, start=1):
        print(f"\nEsperimento {i}:")
        print("y_test:", y_test)
        print("preds:", preds)
