import pandas as pd
import numpy as np

class Metrics:
    """
    Modella le metriche di validazione del modello
    """
    def __init__(self, true_positive=None, true_negative=None, false_positive=None, false_negative=None, filename=None):
        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative
        self.filename = filename
        
    def confusion_matrix(self):
        """
        Restituisce la matrice di confusione come DataFrame.

        La matrice di confusione è una rappresentazione delle prestazioni del modello di classificazione.
        Contiene i conteggi dei veri negativi, falsi positivi, falsi negativi e veri positivi.

        Returns
        -------
        pd.DataFrame
            Un DataFrame contenente i valori della matrice di confusione.
        """
        if None in [self.true_negative, self.false_positive, self.false_negative, self.true_positive]:
            raise ValueError("Tutti i valori della matrice di confusione devono essere forniti.")
        
        return pd.DataFrame({
            'Predicted Negative': [self.true_negative, self.false_negative],
            'Predicted Positive': [self.false_positive, self.true_positive]
        }, index=['Actual Negative', 'Actual Positive'])
    
    def accuracy(self, confusion_matrix=None, K=None):
        """
        Calcola l'accuratezza

        Parameters
        ----------
        confusion_matrix : pd.DataFrame o list di pd.DataFrame, optional
            La matrice di confusione o una lista di matrici di confusione. Se non specificato, si utilizzano i valori dell'istanza.

        K : int, optional
            Il numero di esperimenti. Se non specificato, viene calcolato automaticamente.

        Returns
        -------
        accuracy : float
            L'accuratezza media

        accuracy_scores : list
            I valori di accuratezza per ogni esperimento
        """
        if confusion_matrix is None:
            raise ValueError("La matrice di confusione deve essere fornita.")

        accuracy_scores = []

        # Se la confusion matrix è una lista di DataFrame (per K esperimenti)
        if isinstance(confusion_matrix, list) and isinstance(confusion_matrix[0], pd.DataFrame):
            if K is None:
                K = len(confusion_matrix)  # Calcola automaticamente il numero di esperimenti
            for i in range(K):
                # Calcolo dell'accuratezza
                accuracy_scores.append(float(np.diag(confusion_matrix[i]).sum() / confusion_matrix[i].values.sum()))
        # Se la confusion matrix è un singolo DataFrame
        elif isinstance(confusion_matrix, pd.DataFrame):
            # Calcolo dell'accuratezza
            accuracy_scores.append(float(np.diag(confusion_matrix).sum() / confusion_matrix.values.sum()))
        else:
            raise ValueError("La matrice di confusione deve essere un DataFrame o una lista di DataFrame.")

        # Calcola l'accuratezza media
        accuracy = float(np.mean(accuracy_scores))
        return accuracy, accuracy_scores
    