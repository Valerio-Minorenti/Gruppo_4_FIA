import unittest
import pandas as pd
from Scaling_dati import Scalingdf, GestisciScaling 
class TestScalingdf(unittest.TestCase):
    
    def setUp(self):
        """Inizializza un DataFrame di test."""
        self.df = pd.DataFrame({
            'Sample id number': [1, 2, 3, 4, 5],
            'Feature1': [10, 20, 30, 40, 50],
            'Feature2': [5, 15, 25, 35, 45]
        })
    
    def test_standardizza(self):
        """Testa la standardizzazione del DataFrame."""
        df_standardizzato = Scalingdf.standardizza(self.df)
        self.assertAlmostEqual(df_standardizzato['Feature1'].mean(), 0, places=5)
        self.assertAlmostEqual(df_standardizzato['Feature1'].std(), 1, places=5)
        self.assertAlmostEqual(df_standardizzato['Feature2'].mean(), 0, places=5)
        self.assertAlmostEqual(df_standardizzato['Feature2'].std(), 1, places=5)
        print("\nDataFrame standardizzato:")
        print(df_standardizzato)
    
    def test_normalizza(self):
        """Testa la normalizzazione del DataFrame."""
        df_normalizzato = Scalingdf.normalizza(self.df)
        self.assertAlmostEqual(df_normalizzato['Feature1'].min(), 0, places=5)
        self.assertAlmostEqual(df_normalizzato['Feature1'].max(), 1, places=5)
        self.assertAlmostEqual(df_normalizzato['Feature2'].min(), 0, places=5)
        self.assertAlmostEqual(df_normalizzato['Feature2'].max(), 1, places=5)
        print("\nDataFrame normalizzato:")
        print(df_normalizzato)
    
    def test_scale_features(self):
        """Testa la funzione scale_features per entrambe le strategie."""
        df_standardizzato = GestisciScaling.scale_features('standardizza', self.df)
        self.assertAlmostEqual(df_standardizzato['Feature1'].mean(), 0, places=5)
        self.assertAlmostEqual(df_standardizzato['Feature1'].std(), 1, places=5)
        
        df_normalizzato = GestisciScaling.scale_features('normalizza', self.df)
        self.assertAlmostEqual(df_normalizzato['Feature1'].min(), 0, places=5)
        self.assertAlmostEqual(df_normalizzato['Feature1'].max(), 1, places=5)
        
    def test_scale_features_invalid_strategy(self):
        """Testa che venga sollevato un errore per strategia non valida."""
        with self.assertRaises(ValueError):
            GestisciScaling.scale_features('non_valida', self.df)

if __name__ == '__main__':
    unittest.main()
