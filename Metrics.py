import pandas as pd
import numpy as np

class Metrics:
    """
    Modella le metriche di validazione del modello.
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
        Calcola l'accuratezza.

        Parameters
        ----------
        confusion_matrix : pd.DataFrame o list di pd.DataFrame, optional
            La matrice di confusione o una lista di matrici di confusione. Se non specificato, si utilizzano i valori dell'istanza.

        K : int, optional
            Il numero di esperimenti. Se non specificato, viene calcolato automaticamente.

        Returns
        -------
        accuracy : float
            L'accuratezza media.

        accuracy_scores : list
            I valori di accuratezza per ogni esperimento.
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
    
    def error_rate(self, confusion_matrix=None, K=None):
        """
        Calcola l'error rate.

        Parameters
        ----------
        confusion_matrix : list o pd.DataFrame
            La matrice di confusione o una lista di matrici di confusione.

        K : int, optional
            Il numero di esperimenti. Se non specificato, viene calcolato automaticamente come la lunghezza della lista di matrici di confusione.

        Returns
        -------
        error_rate : float
            Il tasso di errore medio.

        error_rate_scores : list
            I valori del tasso di errore per ogni esperimento.
        """
        error_rate_scores = []

        # Se la confusion matrix è una lista di DataFrame (per K esperimenti)
        if isinstance(confusion_matrix, list) and all(isinstance(cm, pd.DataFrame) for cm in confusion_matrix):
            if K is None:
                K = len(confusion_matrix)
            for i in range(K):
                cm = confusion_matrix[i]
                total = cm.values.sum()
                error_rate_scores.append(1 - np.diag(cm).sum() / total)

        # Se la confusion matrix è un singolo DataFrame
        elif isinstance(confusion_matrix, pd.DataFrame):
            total = confusion_matrix.values.sum()
            error_rate_scores.append(1 - np.diag(confusion_matrix).sum() / total)
        else:
            raise ValueError("La matrice di confusione deve essere un DataFrame o una lista di DataFrame.")

        # Calcola l'error rate medio
        error_rate = float(np.mean(error_rate_scores))

        return error_rate, error_rate_scores
    
    def sensitivity(self, confusion_matrix=None, K=None):
        """
        Calcola la sensitivity (recall) K volte

        Parameters
        ----------
        confusion_matrix : pd.DataFrame o list di pd.DataFrame, optional
        La matrice di confusione o una lista di matrici di confusione. Se non specificato, si utilizzano i valori dell'istanza.

         K : int, optional
        Il numero di esperimenti. Se non specificato, viene calcolato automaticamente.

         Returns
        -------
        sensitivity : float
        La sensitivity media

        sensitivity_scores : list
        I valori della sensitivity per ogni esperimento
        """
        if confusion_matrix is None:
            raise ValueError("La matrice di confusione deve essere fornita.")

        sensitivity_scores = []

        # Se la confusion matrix è una lista di DataFrame (per K esperimenti)
        if isinstance(confusion_matrix, list) and isinstance(confusion_matrix[0], pd.DataFrame):
            if K is None:
                K = len(confusion_matrix)  # Calcola automaticamente il numero di esperimenti
            for i in range(K):
                tp = confusion_matrix[i].loc['Actual Positive', 'Predicted Positive']
                fn = confusion_matrix[i].loc['Actual Positive', 'Predicted Negative']
                sensitivity_scores.append(float(tp / (tp + fn)))
        # Se la confusion matrix è un singolo DataFrame
        elif isinstance(confusion_matrix, pd.DataFrame):
            tp = confusion_matrix.loc['Actual Positive', 'Predicted Positive']
            fn = confusion_matrix.loc['Actual Positive', 'Predicted Negative']
            sensitivity_scores.append(float(tp / (tp + fn)))
        else:
            raise ValueError("La matrice di confusione deve essere un DataFrame o una lista di DataFrame.")

        # Calcola la sensitivity media
        sensitivity = float(np.mean(sensitivity_scores))
        return sensitivity, sensitivity_scores
    

    def specificity(self, confusion_matrix=None, K=None):
        """
        Calcola la specificity K volte

        Parameters
        ----------
        confusion_matrix : pd.DataFrame o list di pd.DataFrame, optional
            La matrice di confusione o una lista di matrici di confusione. Se non specificato, si utilizzano i valori dell'istanza.

        K : int, optional
            Il numero di esperimenti. Se non specificato, viene calcolato automaticamente.

        Returns
        -------
        specificity : float
            La specificity media

        specificity_scores : list
            I valori della specificity per ogni esperimento
        """
        if confusion_matrix is None:
            raise ValueError("La matrice di confusione deve essere fornita.")

        specificity_scores = []

        # Se la confusion matrix è una lista di DataFrame (per K esperimenti)
        if isinstance(confusion_matrix, list) and isinstance(confusion_matrix[0], pd.DataFrame):
            if K is None:
                K = len(confusion_matrix)  # Calcola automaticamente il numero di esperimenti
            for i in range(K):
                tn = confusion_matrix[i].loc['Actual Negative', 'Predicted Negative']
                fp = confusion_matrix[i].loc['Actual Negative', 'Predicted Positive']
                specificity_scores.append(float(tn / (tn + fp)))
        # Se la confusion matrix è un singolo DataFrame
        elif isinstance(confusion_matrix, pd.DataFrame):
            tn = confusion_matrix.loc['Actual Negative', 'Predicted Negative']
            fp = confusion_matrix.loc['Actual Negative', 'Predicted Positive']
            specificity_scores.append(float(tn / (tn + fp)))
        else:
            raise ValueError("La matrice di confusione deve essere un DataFrame o una lista di DataFrame.")

        # Calcola la specificity media
        specificity = float(np.mean(specificity_scores))
        return specificity, specificity_scores
    
    def geometric_mean(self, sensitivity_scores=None, specificity_scores=None, K=None):
        """ 
        Calcola la media geometrica K volte utilizzando valori pre-calcolati di Sensitivity e Specificity.

        Parameters
        ----------
        sensitivity_scores : list of float
        Lista dei valori di sensitivity per ogni esperimento.

        specificity_scores : list of float
        Lista dei valori di specificity per ogni esperimento.

        K : int, optional
        Il numero di esperimenti. Se non specificato, viene calcolato automaticamente.

        Returns
        -------
        gmean : float
        La media geometrica media.

        gmean_scores : list
        Lista dei valori di media geometrica per ogni esperimento.
        """
        if sensitivity_scores is None or specificity_scores is None:
            raise ValueError("Le liste di sensitivity e specificity devono essere fornite.")

        if not isinstance(sensitivity_scores, list) or not isinstance(specificity_scores, list):
            raise ValueError("Sensitivity e specificity devono essere liste di float.")

        if K is None:
            K = len(sensitivity_scores)

        if len(sensitivity_scores) != len(specificity_scores):
            raise ValueError("Le liste di sensitivity e specificity devono avere la stessa lunghezza.")

        # Calcolo del G-Mean per ogni fold
        gmean_scores = [float((sensitivity_scores[i] * specificity_scores[i])) for i in range(K)]

        # Calcolo del G-Mean medio
        average_gmean = float(np.mean(gmean_scores))
        return average_gmean, gmean_scores


    def AUC(self, sensitivity_list, specificity_list, K=None):
        """
        Calcola l'area sotto la curva ROC (AUC) K volte utilizzando Sensitivity e Specificity già calcolate.

        Parameters
        ----------
        sensitivity_list : list of float
        Lista contenente i valori di Sensitivity (TPR) per ogni esperimento.

        specificity_list : list of float
        Lista contenente i valori di Specificity per ogni esperimento.

        K : int, optional
        Il numero di esperimenti. Se non specificato, viene calcolato automaticamente.

        Returns
        -------
        average_auc : float
        L'area media sotto la curva ROC (AUC).

        auc_scores : list
        Lista dei valori di AUC per ogni esperimento.
        """
        if not sensitivity_list or not specificity_list:
            raise ValueError("Le liste di Sensitivity e Specificity devono essere fornite.")

        if K is None:
            K = len(sensitivity_list)

        if len(sensitivity_list) != len(specificity_list):
            raise ValueError("Le liste di Sensitivity e Specificity devono avere la stessa lunghezza.")

        # Convertiamo Specificity in FPR
        fpr_list = [1 - spec for spec in specificity_list]

        auc_scores = []

        # Calcoliamo l'AUC per ogni esperimento
        for i in range(K):
            tpr = sensitivity_list[i]
            fpr = fpr_list[i]

            # Poiché abbiamo un solo punto (TPR, FPR), l'AUC sarà il valore assoluto della differenza tra TPR e FPR
            auc = 0.5 * (1 + tpr - fpr)  # Approccio semplificato usando solo un punto ROC
            auc_scores.append(auc)

        # Calcola l'AUC media
        average_auc = float(np.mean(auc_scores))
        return average_auc, auc_scores