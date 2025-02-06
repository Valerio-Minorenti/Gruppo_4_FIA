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
            if y_test is not None and predicted_proba is not None:
                try:
                    metrics["Area Under Curve"] = self._area_under_curve_knn(y_test, predicted_proba)
                except Exception as e:
                    print(f"Errore durante il calcolo dell'AUC: {e}")
                    metrics["Area Under Curve"] = float('nan')
            else:
                print("Avviso: non sono stati forniti i valori necessari per il calcolo dell'AUC.")
                metrics["Area Under Curve"] = float('nan')

        return metrics

    def calcola_e_stampa_metriche(self, results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                  metrics_to_calculate: List[str]) -> Dict[str, List[float]]:
        """
        Calcola e stampa le metriche per tutti gli esperimenti e restituisce i risultati.

        :param results: Lista di tuple contenenti (y_test, predictions, predicted_proba).
        :param metrics_to_calculate: Lista delle metriche da calcolare.
        :return: Dizionario con i valori delle metriche per ogni esperimento.
        """
        metrics_by_experiment = {metric: [] for metric in metrics_to_calculate}

        try:
            for experiment_index, (y_test, predictions, predicted_proba) in enumerate(results):
                print(f"\nEsperimento {experiment_index + 1} - Calcolo metriche:")
                print("y_test:", y_test)
                print("y_pred:", predictions)

                # Calcolo delle metriche
                tp, tn, fp, fn = self.confu(y_test, predictions)
                metrics_calc = MetricsCalculator(true_positive=tp, true_negative=tn, false_positive=fp,
                                                 false_negative=fn)
                cm = metrics_calc.confusion_matrix()

                print("\nMatrice di confusione:")
                print(cm)

                # Calcolo delle metriche, inclusa l'AUC
                metrics = metrics_calc.calculate_metrics(
                    confusion_matrix=cm,
                    metrics_to_calculate=metrics_to_calculate,
                    y_test=y_test,
                    predictions=predictions,
                    predicted_proba=predicted_proba
                )

                # Aggiungi i risultati per ogni metrica
                for metric, value in metrics.items():
                    metrics_by_experiment[metric].append(value)

                # Stampa delle metriche calcolate
                print("\nMetriche calcolate (singolo esperimento):")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}" if not np.isnan(value) else f"{metric}: N/A")

        except ValueError as ve:
            print(f"Errore nei dati di input o nei parametri: {ve}")
        except Exception as e:
            print(f"Si è verificato un errore durante il calcolo delle metriche: {e}")

        # Calcolo della media delle metriche su K esperimenti
        avg_metrics = {metric: np.mean(values) for metric, values in metrics_by_experiment.items()}
        print("\nMedia delle metriche su tutti gli esperimenti:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Stampa dei risultati delle metriche (lista per ogni esperimento)
        print("\nValori delle metriche per tutti gli esperimenti:")
        for metric, values in metrics_by_experiment.items():
            print(f"{metric}: {values}")

        return metrics_by_experiment

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
        """
        y_test = np.asarray(y_test)
        predicted_proba = np.asarray(predicted_proba)
        positive_label = 2

        if len(np.unique(y_test)) < 2:
            print("Errore: Actual values contiene solo una classe, AUC non può essere calcolata!")
            return 0.0

        sorted_indices = np.argsort(predicted_proba)
        y_true_sorted = y_test[sorted_indices]

        total_positives = np.sum(y_true_sorted == positive_label)
        total_negatives = np.sum(y_true_sorted != positive_label)

        if total_positives == 0 or total_negatives == 0:
            print("Errore: non ci sono abbastanza esempi per calcolare la curva ROC.")
            return 0.0

        TPR = np.cumsum(y_true_sorted == positive_label) / total_positives
        FPR = np.cumsum(y_true_sorted != positive_label) / total_negatives

        auc =np.trapz(TPR, FPR)
        return float(auc)
