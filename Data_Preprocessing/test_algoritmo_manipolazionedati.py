import unittest
import pandas as pd
import os
from Data_Preprocessing.Algoritmo_manipolazione_dati import ManipolaDati


class TestManipolaDati(unittest.TestCase):

    def setUp(self):
        self.input_path = "test_input.csv"
        self.output_path = "test_output.csv"
        self.manipolatore = ManipolaDati(self.input_path, self.output_path)

        data = {
            "ID": [1, 2, 3, 4, 5],
            "A": [10, 20, 30, 40, 50],
            "B": ["a", "b", "c", "d", "e"],
            "Duplicati": [1, 1, 2, 2, 3]
        }
        self.df = pd.DataFrame(data)

    def test_salva_percorso(self):
        self.assertEqual(self.manipolatore.salva_percorso(), (self.input_path, self.output_path))

    def test_elimina_colonne(self):
        df_risultato = self.manipolatore.elimina_colonne(self.df, ["A", "B"])
        self.assertNotIn("A", df_risultato.columns)
        self.assertNotIn("B", df_risultato.columns)

    def test_ordina_colonna(self):
        df_risultato = self.manipolatore.ordina_colonna(self.df, "A")
        self.assertTrue(df_risultato["A"].is_monotonic_increasing)

    def test_elimina_duplicati_su_colonna(self):
        df_risultato = self.manipolatore.elimina_duplicati_su_colonna(self.df, "Duplicati")
        self.assertEqual(len(df_risultato), len(df_risultato["Duplicati"].unique()))

    def test_converti_a_numerico(self):
        df_risultato = self.manipolatore.converti_a_numerico(self.df.copy())
        self.assertTrue(pd.api.types.is_integer_dtype(df_risultato["A"]))

    def test_correggi_valori(self):
        df = pd.DataFrame({"Valori": [0, 5, 10, 15, 20]})
        df_risultato = self.manipolatore.correggi_valori(df)
        expected = [1, 5, 10, 9, 2]
        self.assertListEqual(df_risultato["Valori"].tolist(), expected)

    def test_correggi_class_type(self):
        df = pd.DataFrame({"ClassType": [1, 2, 1, 3, 1]})
        df_risultato = self.manipolatore.correggi_class_type(df, ["ClassType"])
        self.assertFalse((df_risultato["ClassType"] == 1).any())

    def test_scambia_colonne(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]})
        df_risultato = self.manipolatore.scambia_colonne(df, "A", "B", "C", "D")
        self.assertListEqual(df_risultato.columns.tolist(), ["B", "A", "D", "C"])

    def test_salva_file(self):
        self.manipolatore.salva_file(self.df, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        os.remove(self.output_path)


if __name__ == "__main__":
        unittest.main()