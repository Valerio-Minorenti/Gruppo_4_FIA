import numpy as np
import pandas as pd
from kNN_classifier import KNN  # Assicurati che la classe KNN sia definita correttamente

class StratifiedCrossValidation:
    """
    Classe per eseguire stratified cross-validation.
    """

    def __init__(self, K):
        """
        Costruttore della classe StratifiedCrossValidation.
        :param K: Numero di fold per la cross-validation.
        """
        self.K = K  # Numero di fold

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
    
    def run_experiments(self, df, class_column, k_neighbors):
        """
        Esegue la stratified cross-validation per un numero specificato di esperimenti e calcola le accuratezze.

        :param df: DataFrame contenente i dati.
        :param class_column: Nome della colonna che contiene le etichette di classe.
        :param k_neighbors: Numero di vicini da considerare per il classificatore KNN.
        :return: Lista delle accuratezze per ciascun esperimento.
        """
        accuracies = []  # Inizializza una lista vuota per memorizzare le accuratezze di ciascun esperimento
        folds = self.split(df, class_column)  # Divide il DataFrame in set di training e test
        for train_set, test_set in folds:
            X_train = train_set.drop(columns=class_column).values
            y_train = train_set[class_column].values
            X_test = test_set.drop(columns=class_column).values
            y_test = test_set[class_column].values

            classifier = KNN(k=k_neighbors)  # Crea un'istanza del classificatore KNN con il numero di vicini specificato
            classifier.fit(X_train, y_train)  # Addestra il classificatore sui dati di addestramento
            predictions = classifier.predict(X_test)  # Predice le etichette per i dati di test
            accuracy = np.mean(predictions == y_test)  # Calcola l'accuratezza delle predizioni
            accuracies.append(float(accuracy))  # Aggiunge l'accuratezza alla lista, convertendola in float per evitare np.float64

        return accuracies  # Restituisce la lista delle accuratezze

if __name__ == '__main__':
    # Creazione di un DataFrame di esempio
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 8],
        'feature2': [0.5, 4.3, 3.3, 4.0, 2.5, 5.7, 5.3, 7.4, 8.1],
        'Class': [2, 4, 4, 4, 2, 2, 4, 4, 4]
    }
    df_example = pd.DataFrame(data)
    print("DataFrame di esempio:")
    print(df_example)

    # Chiediamo all'utente di specificare il numero di fold e il numero di vicini
    K = int(input("\nInserisci il numero di fold K (ad esempio, 3): "))
    k_neighbors = 3;#int(input("Inserisci il numero di vicini (k) per KNN: "))

    # Istanziazione della classe StratifiedCrossValidation
    splitter = StratifiedCrossValidation(K=K)

    # Eseguiamo lo split stratificato
    folds = splitter.split(df_example, class_column="Class")

    # Stampa dei risultati
    for fold_index, (train_set, test_set) in enumerate(folds):
        print(f"\nFold {fold_index + 1}:")
        print("Train set:")
        print(train_set)
        print("\nTest set:")
        print(test_set)

    # Eseguiamo gli esperimenti stratificati
    accuracies = splitter.run_experiments(df_example, class_column="Class", k_neighbors=k_neighbors)

    # Stampa dei risultati degli esperimenti
    for experiment_index, accuracy in enumerate(accuracies):
        print(f"Esperimento {experiment_index + 1} - Accuratezza: {accuracy:.2f}")