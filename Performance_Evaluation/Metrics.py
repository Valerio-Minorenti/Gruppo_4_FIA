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
        """
        unique_classes = np.unique(y_test)
        if len(unique_classes) != 2:
            raise ValueError("Il calcolo della matrice di confusione richiede due classi distinte.")

        positive_class = max(unique_classes)
        negative_class = min(unique_classes)

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

    def calculate_metrics(self,
                      confusion_matrix: pd.DataFrame,
                      metrics_to_calculate: List[str],
                      y_test: np.ndarray = None,
                      predictions: np.ndarray = None,
                      predicted_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calcola le metriche specificate da una matrice di confusione.
        """
        tp = confusion_matrix.loc['Actual Positive', 'Predicted Positive']
        tn = confusion_matrix.loc['Actual Negative', 'Predicted Negative']
        fp = confusion_matrix.loc['Actual Negative', 'Predicted Positive']
        fn = confusion_matrix.loc['Actual Positive', 'Predicted Negative']

        metrics = {}
        if "Accuracy Rate" in metrics_to_calculate:
            metrics["Accuracy Rate"] = self._accuracy_rate(tp, tn, fp, fn)
        if "Error Rate" in metrics_to_calculate:
            metrics["Error Rate"] = self._error_rate(tp, tn, fp, fn)
        if "Sensitivity" in metrics_to_calculate:
            metrics["Sensitivity"] = self._sensitivity(tp, fn)
        if "Specificity" in metrics_to_calculate:
            metrics["Specificity"] = self._specificity(tn, fp)
        if "Geometric Mean" in metrics_to_calculate:
            metrics["Geometric Mean"] = self._geometric_mean(tp, tn, fp, fn)
        if "Area Under Curve" in metrics_to_calculate:
            if y_test is not None and predictions is not None:
                # Calcolo dell'AUC usando i punteggi e y_test
                try:
                    metrics["Area Under Curve"] = self._area_under_curve_knn(y_test, predictions)
                except Exception as e:
                    print(f"Errore durante il calcolo dell'AUC: {e}")
                    metrics["Area Under Curve"] = float('nan')
            else:
                print("Avviso: non sono stati forniti i valori necessari per il calcolo dell'AUC.")
                metrics["Area Under Curve"] = float('nan')

        return metrics

    def _accuracy_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        total = tp + tn + fp + fn
        return float((tp + tn) / total if total > 0 else 0.0)

    def _error_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        total = tp + tn + fp + fn
        return float((fp + fn) / total if total > 0 else 0.0)

    def _sensitivity(self, tp: int, fn: int) -> float:
        actual_positive = tp + fn
        return float(tp / actual_positive if actual_positive > 0 else 0.0)

    def _specificity(self, tn: int, fp: int) -> float:
        actual_negative = tn + fp
        return float(tn / actual_negative if actual_negative > 0 else 0.0)

    def _geometric_mean(self, tp: int, tn: int, fp: int, fn: int) -> float:
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        return float(np.sqrt(sensitivity * specificity))

    def _area_under_curve_knn(self, y_test: np.ndarray, predicted_proba: np.ndarray) -> float:
        """
        Calcola l'Area Under Curve (AUC) usando i valori di probabilità predetti.

        :param y_test: Array dei valori reali.
        :param predicted_proba: Array dei valori di probabilità predetti per la classe positiva.
        :return: Valore dell'AUC.
        """
        # Converto esplicitamente gli input in array NumPy, se necessario
        y_test = np.asarray(y_test)
        predicted_proba = np.asarray(predicted_proba)

        # Classe positiva (puoi cambiare il valore se necessario)
        positive_label = 2

        # Controlla se ci sono almeno due classi nei valori reali
        if len(np.unique(y_test)) < 2:
            print("Errore: Actual values contiene solo una classe, AUC non può essere calcolata!")
            return 0.0

        # Ordina i valori delle probabilità predette e i corrispondenti valori reali
        sorted_indices = np.argsort(predicted_proba)
        y_true_sorted = y_test[sorted_indices]

        # Calcola TPR e FPR
        total_positives = np.sum(y_true_sorted == positive_label)
        total_negatives = np.sum(y_true_sorted != positive_label)

        if total_positives == 0 or total_negatives == 0:
            print("Errore: non ci sono abbastanza esempi per calcolare la curva ROC.")
            return 0.0

        # Calcolo dei True Positive Rate (TPR) e False Positive Rate (FPR)
        TPR = np.cumsum(y_true_sorted == positive_label) / total_positives
        FPR = np.cumsum(y_true_sorted != positive_label) / total_negatives

        # Calcolo dell'AUC usando il metodo del trapezio
        auc = np.trapz(TPR, FPR)

        return float(auc)

    @staticmethod
    def scegli_metriche() -> List[str]:
        print("Scegli le metriche da calcolare:")
        print("1. Accuracy Rate")
        print("2. Error Rate")
        print("3. Sensitivity")
        print("4. Specificity")
        print("5. Geometric Mean")
        print("6. Area Under Curve")
        print("7. Calcola tutte le metriche")
        metrics_choice = input("Inserisci i numeri delle metriche da calcolare separati da virgola (es. 1,2,3): ").split(',')

        if "7" in metrics_choice:
            return ["Accuracy Rate", "Error Rate", "Sensitivity", "Specificity", "Geometric Mean", "Area Under Curve"]

        metrics_map = {
            "1": "Accuracy Rate",
            "2": "Error Rate",
            "3": "Sensitivity",
            "4": "Specificity",
            "5": "Geometric Mean",
            "6": "Area Under Curve"
        }

        return [metrics_map[choice.strip()] for choice in metrics_choice if choice.strip() in metrics_map]
