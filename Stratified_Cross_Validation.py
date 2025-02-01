import numpy as np
import pandas as pd
from kNN_classifier import KNN

class StratifiedCrossValidation:
    """
    Classe per eseguire stratified cross-validation.
    """

    def __init__(self, K, df, class_column, k_neighbors):
        self.K = K  # Numero di fold
        self.df = pd.DataFrame(df)
        self.class_column = class_column
        self.k_neighbors = k_neighbors

    def split(self, df, class_column):
        df["fold"] = np.nan

        # Distribuisce le righe ai fold mantenendo le proporzioni delle classi
        for label in df[class_column].unique():
            class_indices = df[df[class_column] == label].index.tolist()

            # Determina quanti campioni per fold
            fold_sizes = [len(class_indices) // self.K for _ in range(self.K)]
            for i in range(len(class_indices) % self.K):
                fold_sizes[i] += 1

            current = 0
            for i in range(self.K):
                start, stop = current, current + fold_sizes[i]
                df.loc[class_indices[start:stop], "fold"] = i
                current = stop

        folds = []
        for i in range(self.K):
            test_set = df[df["fold"] == i].drop(columns="fold")
            train_set = df[df["fold"] != i].drop(columns="fold")
            folds.append((train_set, test_set))

        return folds

    def run_experiments(self):
        results = []
        folds = self.split(self.df, self.class_column)

        for train_set, test_set in folds:
            X_train = train_set.drop(columns=self.class_column).values
            y_train = train_set[self.class_column].values
            X_test = test_set.drop(columns=self.class_column).values
            y_test = test_set[self.class_column].values

            classifier = KNN(k=self.k_neighbors)
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)

            results.append((y_test, predictions))

        return results

if __name__ == '__main__':
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 8],
        'feature2': [0.5, 4.3, 3.3, 4.0, 2.5, 5.7, 5.3, 7.4, 8.1],
        'Class': [2, 4, 4, 4, 2, 2, 4, 4, 4]
    }
    df_example = pd.DataFrame(data)
    print("DataFrame di esempio:")
    print(df_example)

    K = int(input("\nInserisci il numero di fold K (ad esempio, 3): "))
    k_neighbors = 3

    splitter = StratifiedCrossValidation(K=K, df=df_example, class_column="Class", k_neighbors=k_neighbors)

    results = splitter.run_experiments()

    # Stampa delle etichette reali e delle predizioni per ciascun esperimento
    for experiment_index, (y_test, predictions) in enumerate(results):
        print(f"\nEsperimento {experiment_index + 1}:")
        print("Etichette reali (y_test):", [int(val) for val in y_test])
        print("Predizioni (predictions):", [int(val) for val in predictions])
