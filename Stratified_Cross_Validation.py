import numpy as np
import pandas as pd
from kNN_classifier import KNN

class StratifiedCrossValidation:
    """
    Classe per eseguire stratified cross-validation.
    """

    def __init__(self, K, df, class_column, k_neighbors):
        """
        Costruttore della classe StratifiedCrossValidation.
        :param K: Numero di fold per la cross-validation.
        :param df: DataFrame contenente i dati.
        :param class_column: Nome della colonna che contiene le etichette di classe.
        :param k_neighbors: Numero di vicini da considerare per il classificatore KNN.
        """
        self.K = K  # Numero di fold
        self.df = pd.DataFrame(df)
        self.class_column = class_column
        self.k_neighbors = k_neighbors

    def split(self, df, class_column):
        """
        Esegue la stratified cross-validation sul DataFrame.
        :param df: DataFrame da dividere.
        :param class_column: Nome della colonna che contiene le etichette di classe.
        :return: Lista di tuple (train_set, test_set) per ogni fold.
        """
        # Aggiunta della colonna 'fold' per assegnare le righe ai diversi fold
        df["fold"] = np.nan  # Ogni riga appartiene a un solo fold per iterazione.

        # Distribuisce le righe ai fold mantenendo le proporzioni delle classi
        for label in df[class_column].unique():
            class_indices = df[df[class_column] == label].index.tolist()

            # Determina quanti campioni per fold
            fold_sizes = [len(class_indices) // self.K for _ in range(self.K)]
            for i in range(len(class_indices) % self.K):
                fold_sizes[i] += 1

            # Assegna gli indici ai fold
            current = 0
            for i in range(self.K):
                start, stop = current, current + fold_sizes[i]
                df.loc[class_indices[start:stop], "fold"] = i
                current = stop

        # Crea i set di training e test per ogni fold
        folds = []
        for i in range(self.K):
            test_set = df[df["fold"] == i].drop(columns="fold")
            train_set = df[df["fold"] != i].drop(columns="fold")
            folds.append((train_set, test_set))

        return folds
    
    def run_experiments(self):
        """
        Esegue la stratified cross-validation per un numero specificato di esperimenti e calcola le accuratezze.
        :return: Lista delle accuratezze per ciascun esperimento.
        """
        accuracies = []  # Inizializza una lista vuota per memorizzare le accuratezze di ciascun esperimento
        folds = self.split(self.df, self.class_column)  # Divide il DataFrame in set di training e test
        for train_set, test_set in folds:
            X_train = train_set.drop(columns=self.class_column).values
            y_train = train_set[self.class_column].values
            X_test = test_set.drop(columns=self.class_column).values
            y_test = test_set[self.class_column].values

            classifier = KNN(k=self.k_neighbors)  # Crea un'istanza del classificatore KNN con il numero di vicini specificato
            classifier.fit(X_train, y_train)  # Addestra il classificatore sui dati di addestramento
            predictions = classifier.predict(X_test)  # Predice le etichette per i dati di test
            # Stampa dei risultati per ogni fold
            print("Stratified Cross Validation eseguito con successo.")
            print(f"Numero di campioni nel test set: {len(y_test)}")
            print(f"Esempio di etichette reali: {y_test[:10]}")
            print(f"Esempio di etichette predette: {[float(pred) for pred in predictions[:10]]}")

