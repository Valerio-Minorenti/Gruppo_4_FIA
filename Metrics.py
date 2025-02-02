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

    def calculate_metrics(self, confusion_matrix: pd.DataFrame, metrics_to_calculate: List[str]) -> Dict[str, float]:
        """
        Calcola le metriche specificate da una matrice di confusione.
        
        Parameters
        ----------
        confusion_matrix : pd.DataFrame
            La matrice di confusione.
        metrics_to_calculate : List[str]
            Lista delle metriche da calcolare.

        Returns
        -------
        Dict[str, float]
            Dizionario contenente le metriche calcolate.
        """
        tp = confusion_matrix.loc['Actual Positive', 'Predicted Positive']
        tn = confusion_matrix.loc['Actual Negative', 'Predicted Negative']
        fp = confusion_matrix.loc['Actual Negative', 'Predicted Positive']
        fn = confusion_matrix.loc['Actual Positive', 'Predicted Negative']

        metrics = {}
        if "Accuracy Rate" in metrics_to_calculate or "all" in metrics_to_calculate:
            metrics["Accuracy Rate"] = self._accuracy_rate(tp, tn, fp, fn)
        if "Error Rate" in metrics_to_calculate or "all" in metrics_to_calculate:
            metrics["Error Rate"] = self._error_rate(tp, tn, fp, fn)
        if "Sensitivity" in metrics_to_calculate or "all" in metrics_to_calculate:
            metrics["Sensitivity"] = self._sensitivity(tp, fn)
        if "Specificity" in metrics_to_calculate or "all" in metrics_to_calculate:
            metrics["Specificity"] = self._specificity(tn, fp)
        if "Geometric Mean" in metrics_to_calculate or "all" in metrics_to_calculate:
            metrics["Geometric Mean"] = self._geometric_mean(tp, tn, fp, fn)
        if "Area Under Curve" in metrics_to_calculate or "all" in metrics_to_calculate:
            metrics["Area Under Curve"] = self._area_under_curve(tp, tn, fp, fn)
        
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

    @staticmethod
    def scegli_metriche() -> List[str]:
        """
        Permette all'utente di scegliere le metriche da calcolare.
        
        Returns
        -------
        List[str]
            Lista delle metriche scelte dall'utente.
        """
        print("Scegli le metriche da calcolare:")
        print("1. Accuracy Rate")
        print("2. Error Rate")
        print("3. Sensitivity")
        print("4. Specificity")
        print("5. Geometric Mean")
        print("6. Area Under Curve")
        print("7. Calcola tutte le metriche")
        metrics_choice = input("Inserisci i numeri delle metriche da calcolare separati da virgola (es. 1,2,3): ").split(',')

        # Mappa dei numeri alle metriche
        metrics_map = {
            "1": "Accuracy Rate",
            "2": "Error Rate",
            "3": "Sensitivity",
            "4": "Specificity",
            "5": "Geometric Mean",
            "6": "Area Under Curve",
            "7": "all"
        }

        # Converti le scelte dell'utente in metriche
        metrics_to_calculate = [metrics_map[choice.strip()] for choice in metrics_choice if choice.strip() in metrics_map]
        return metrics_to_calculate


# Esempio di utilizzo
if __name__ == "__main__":
    y_test = np.array([4, 2, 4, 2])
    predictions = np.array([4, 2, 4, 4])
    tp, tn, fp, fn = MetricsCalculator.confu(y_test, predictions)
    metrics_calc = MetricsCalculator(true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)
    cm = metrics_calc.confusion_matrix()
    print("Confusion Matrix:\n", cm)
    
    # Chiedi all'utente quali metriche calcolare
    metrics_to_calculate = MetricsCalculator.scegli_metriche()
    
    metrics = metrics_calc.calculate_metrics(cm, metrics_to_calculate)
    print("Metrics:\n", metrics)