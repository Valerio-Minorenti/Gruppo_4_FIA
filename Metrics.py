import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

class MetricsCalculator:
    """
    Modella le metriche di validazione del modello.
    """

    def __init__(self, true_positive=None, true_negative=None, false_positive=None, false_negative=None):
        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative

    @staticmethod
    def confu(y_test: np.ndarray, predictions: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calcola i valori della matrice di confusione generica.
        Determina la classe positiva e negativa dai dati stessi.
        """
        unique_classes = np.unique(y_test)
        if len(unique_classes) != 2:
            raise ValueError("Il calcolo della matrice di confusione richiede due classi distinte.")

        # Assegna automaticamente la classe positiva come quella più alta
        positive_class = max(unique_classes)
        negative_class = min(unique_classes)

        # Calcola i valori della matrice di confusione
        tp = np.sum((y_test == positive_class) & (predictions == positive_class))
        tn = np.sum((y_test == negative_class) & (predictions == negative_class))
        fp = np.sum((y_test == negative_class) & (predictions == positive_class))
        fn = np.sum((y_test == positive_class) & (predictions == negative_class))

        return tp, tn, fp, fn

    def confusion_matrix(self) -> pd.DataFrame:
        """
        Restituisce la matrice di confusione come DataFrame.
        """
        if None in [self.true_negative, self.false_positive, self.false_negative, self.true_positive]:
            raise ValueError("Tutti i valori della matrice di confusione devono essere forniti.")
        
        return pd.DataFrame({
            'Predicted Negative': [self.true_negative, self.false_negative],
            'Predicted Positive': [self.false_positive, self.true_positive]
        }, index=['Actual Negative', 'Actual Positive'])

    def calculate_metrics(self, confusion_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calcola tutte le metriche da una matrice di confusione.
        
        Parameters
        ----------
        confusion_matrix : pd.DataFrame
            La matrice di confusione.

        Returns
        -------
        Dict[str, float]
            Dizionario contenente tutte le metriche calcolate.
        """
        tp = confusion_matrix.loc['Actual Positive', 'Predicted Positive']
        tn = confusion_matrix.loc['Actual Negative', 'Predicted Negative']
        fp = confusion_matrix.loc['Actual Negative', 'Predicted Positive']
        fn = confusion_matrix.loc['Actual Positive', 'Predicted Negative']

        metrics = {
            "Accuracy Rate": self._accuracy_rate(tp, tn, fp, fn),
            "Error Rate": self._error_rate(tp, tn, fp, fn),
            "Sensitivity": self._sensitivity(tp, fn),
            "Specificity": self._specificity(tn, fp),
            "Geometric Mean": self._geometric_mean(tp, tn, fp, fn),
            "Area Under Curve": self._area_under_curve(tp, tn, fp, fn)
        }
        return metrics

    def _accuracy_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calcola l'accuracy rate.
        """
        total = tp + tn + fp + fn
        return float((tp + tn) / total) if total > 0 else 0.0

    def _error_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calcola l'error rate.
        """
        total = tp + tn + fp + fn
        return float((fp + fn) / total) if total > 0 else 0.0

    def _sensitivity(self, tp: int, fn: int) -> float:
        """
        Calcola la sensitivity (recall).
        """
        actual_positive = tp + fn
        return float(tp / actual_positive) if actual_positive > 0 else 0.0

    def _specificity(self, tn: int, fp: int) -> float:
        """
        Calcola la specificity (true negative rate).
        """
        actual_negative = tn + fp
        return float(tn / actual_negative) if actual_negative > 0 else 0.0

    def _geometric_mean(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calcola la Geometric Mean (G-Mean).
        """
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        return float(np.sqrt(sensitivity * specificity))

    def _area_under_curve(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calcola l'Area Under the Curve (AUC).
        """
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        return float((sensitivity + specificity) / 2)


# Esempio di utilizzo
if __name__ == "__main__":
    y_test = np.array([4, 2, 4, 2])
    predictions = np.array([4, 2, 4, 4])
    tp, tn, fp, fn = MetricsCalculator.confu(y_test, predictions)
    metrics_calc = MetricsCalculator(true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)
    cm = metrics_calc.confusion_matrix()
    print("Confusion Matrix:\n", cm)
    metrics = metrics_calc.calculate_metrics(cm)
    print("Metrics:\n", metrics)