import unittest
import numpy as np
import pandas as pd
from Performance_Evaluation.Metrics import MetricsCalculator

class TestMetricsCalculator(unittest.TestCase):

    def setUp(self):
        """
        Imposta i dati di test per la matrice di confusione.
        """
        self.y_test = np.array([4, 2, 4, 2, 4, 4, 2, 2])
        self.y_pred = np.array([4, 2, 4, 4, 2, 4, 2, 2])

        self.tp, self.tn, self.fp, self.fn = MetricsCalculator.confu(self.y_test, self.y_pred)

        self.metrics_calc = MetricsCalculator(
            true_positive=self.tp,
            true_negative=self.tn,
            false_positive=self.fp,
            false_negative=self.fn
        )

    def test_confusion_matrix_calculation(self):
        """
        Testa se la funzione confu() calcola correttamente TP, TN, FP, FN.
        """
        expected_tp = np.sum((self.y_test == 4) & (self.y_pred == 4))
        expected_tn = np.sum((self.y_test == 2) & (self.y_pred == 2))
        expected_fp = np.sum((self.y_test == 2) & (self.y_pred == 4))
        expected_fn = np.sum((self.y_test == 4) & (self.y_pred == 2))

        self.assertEqual(self.tp, expected_tp, "Il valore di True Positive è errato.")
        self.assertEqual(self.tn, expected_tn, "Il valore di True Negative è errato.")
        self.assertEqual(self.fp, expected_fp, "Il valore di False Positive è errato.")
        self.assertEqual(self.fn, expected_fn, "Il valore di False Negative è errato.")

    def test_confusion_matrix_dataframe(self):
        """
        Testa se la funzione confusion_matrix() genera un DataFrame corretto.
        """
        cm = self.metrics_calc.confusion_matrix()
        expected_cm = pd.DataFrame({
            'Predicted Negative': [self.tn, self.fn],
            'Predicted Positive': [self.fp, self.tp]
        }, index=['Actual Negative', 'Actual Positive'])

        pd.testing.assert_frame_equal(cm, expected_cm, "La matrice di confusione generata non è corretta.")

    def test_calculate_metrics(self):
        """
        Testa se calculate_metrics() calcola correttamente le metriche principali.
        """
        cm = self.metrics_calc.confusion_matrix()
        metrics = self.metrics_calc.calculate_metrics(cm, ["Accuracy Rate", "Sensitivity", "Specificity"])

        expected_accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        expected_sensitivity = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        expected_specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0

        self.assertAlmostEqual(metrics["Accuracy Rate"], expected_accuracy, places=4, msg="Accuracy Rate errato.")
        self.assertAlmostEqual(metrics["Sensitivity"], expected_sensitivity, places=4, msg="Sensitivity errato.")
        self.assertAlmostEqual(metrics["Specificity"], expected_specificity, places=4, msg="Specificity errato.")

if __name__ == "__main__":
    unittest.main()
