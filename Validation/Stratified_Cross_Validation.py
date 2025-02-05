import numpy as np
import pandas as pd
from kNN_classifier import KNN
from Validation.Validation_Strategy import ValidationStrategy

class StratifiedCrossValidation(ValidationStrategy):
    """
    Classe per eseguire stratified cross-validation.
    Eredita da ValidationStrategy, che richiede un metodo astratto generate_splits().
    """

    def __init__(self, K, df, class_column, k_neighbors):
        """
        Costruttore: inizializza i parametri necessari.

        :param K: Numero di fold (intero).
        :param df: DataFrame contenente le feature e la colonna di classe.
        :param class_column: Nome della colonna che contiene le etichette di classe.
        :param k_neighbors: Numero di vicini per il KNN (oppure parametri simili per un classificatore).
        """
        self.K = K  # Numero di fold
        self.df = pd.DataFrame(df).copy()
        self.class_column = class_column
        self.k_neighbors = k_neighbors

    def generate_splits(self, k=None):
        """
        Metodo astratto richiesto da ValidationStrategy.
        Se 'k' è specificato, sovrascrive temporaneamente self.K.
        Poi richiama la logica di run_experiments, che esegue la cross validation.

        :param k: Numero di fold da utilizzare in questa esecuzione (opzionale).
        :return: Lista di (y_test, predicted_proba) per ogni fold.
        """
        if k is not None:
            self.K = k

        return self.run_experiments()

    def split(self, df, class_column):
        """
        Esegue la suddivisione stratificata del DataFrame in self.K fold,
        restituendo una lista di tuple (train_set, test_set).
        """
        df = df.copy()
        df["fold"] = np.nan

        # Distribuisce le righe nei fold mantenendo le proporzioni delle classi
        for label in df[class_column].unique():
            class_indices = df[df[class_column] == label].index.tolist()
            np.random.shuffle(class_indices)

            # Determina quanti campioni assegnare a ciascun fold
            fold_sizes = [len(class_indices) // self.K for _ in range(self.K)]
            for i in range(len(class_indices) % self.K):
                fold_sizes[i] += 1

            current = 0
            for i in range(self.K):
                start, stop = current, current + fold_sizes[i]
                df.loc[class_indices[start:stop], "fold"] = i
                current = stop

        # Crea la lista di tuple (train_set, test_set)
        folds = []
        for i in range(self.K):
            test_set = df[df["fold"] == i].drop(columns="fold")
            train_set = df[df["fold"] != i].drop(columns="fold")
            folds.append((train_set, test_set))

        return folds

    def run_experiments(self):
        """
        Esegue la stratified cross-validation vera e propria:
          1) suddivide i dati tramite split(),
          2) allena il classificatore KNN su train_set,
          3) calcola le probabilità previste su test_set per ogni fold.

        :return: Lista di tuple (y_test, predicted_proba) per ogni fold.
        """
        results = []
        folds = self.split(self.df, self.class_column)

        for train_set, test_set in folds:
            # Prepara i dati di addestramento e test
            X_train = train_set.drop(columns=self.class_column).values
            y_train = train_set[self.class_column].values
            X_test = test_set.drop(columns=self.class_column).values
            y_test = test_set[self.class_column].values

            # Inizializza e addestra il classificatore KNN
            classifier = KNN(k=self.k_neighbors)
            classifier.fit(X_train, y_train)


            predictions = classifier.predict(X_test)
            predictions = [int(pred) for pred in predictions]

            # Calcola le probabilità previste sul test set
            predicted_proba = classifier.predict_proba(X_test)

            # Converti le probabilità della classe positiva in un array continuo
            positive_class = 4  # La classe positiva cioè MALIGNIO sia '4'
            predicted_proba_continuous = [proba.get(positive_class, 0.0) for proba in predicted_proba]
            predicted_proba_continuous = [float(predprob) for predprob in predicted_proba_continuous]
            # Salvataggio dei risultati di questo fold
            y_test = [int(val) for val in y_test]
            results.append((y_test, predictions, predicted_proba_continuous))

        return results

# Test di esempio
if __name__ == '__main__':
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 8],
        'feature2': [0.5, 4.3, 3.3, 4.0, 2.5, 5.7, 5.3, 7.4, 8.1],
        'Class':    [2,   4,   4,   4,   2,   2,   4,   4,   4]
    }

    df_example = pd.DataFrame(data)
    print("DataFrame di esempio:")
    print(df_example)

    K = int(input("\nInserisci il numero di fold K (ad es. 3): "))
    k_neighbors = 7

    # Istanzia la classe e richiama generate_splits()
    splitter = StratifiedCrossValidation(K=K, df=df_example, class_column="Class", k_neighbors=k_neighbors)
    results = splitter.generate_splits()

    # Stampa delle etichette reali e delle probabilità previste per ogni fold
    for experiment_index, (y_test,pred, predicted_proba) in enumerate(results, start=1):
        print(f"\nFold {experiment_index}:")
        print("Etichette reali (y_test):", [int(val) for val in y_test])
        print("Etichette reali (y_test):", [int(val) for val in pred])
        print("Probabilità previste (predicted_proba):", predicted_proba)
