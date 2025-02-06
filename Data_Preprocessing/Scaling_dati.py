import pandas as pd

class Scalingdf:
    """
    classe per lo scaling, alcune colonne hanno valori assoluti e scale più
    alte rispetto ad altre anche se hanno lo stesso peso.
    """

    @staticmethod
    def standardizza(df: pd.DataFrame) -> pd.DataFrame:
        """
        standardizza tutte le colonne tranne quella degli ID

        Args:
            df (pd.DataFrame): df da scalare

        Returns:
            pd.DataFrame: df con colonne standardizzate
        """
        # copia il df di ingresso senza modifcarlo direttamente
        df_scalato = df.copy()

        # si prendono tutte le colonne tramnne sample id number
        colonne_da_scalare = [col for col in df.columns if col  not in ["Sample id number","classtype_v1"]]

        # formula standardizzazione
        for col in colonne_da_scalare:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:  # Evita la divisione per zero
                df_scalato[col] = (df[col] - mean_val) / std_val
            else:
                df_scalato[col] = 0  # Se std=0, assegna 0

        print("tutte le colonne di intertesse sono state standardizzate.")
        return df_scalato
    @staticmethod
    def normalizza(df: pd.DataFrame) -> pd.DataFrame:
        """
        normalizza tutte le colonne tranne quella degli ID

        Args:
            df (pd.DataFrame): df da scalare

        Returns:
            pd.DataFrame: df con le colonne normalizzate
        """
        # copia il df di ingresso senza modifcarlo direttamente
        df_scalato = df.copy()

        # si prendono tutte le colonne tramnne sample id number
        colonne_da_scalare = [col for col in df.columns if col not in ["Sample id number","classtype_v1"]]

        # formula per normalizzare le colonne
        for col in colonne_da_scalare:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val != 0:  # Evita la divisione per zero
                df_scalato[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_scalato[col] = 0  # Se min == max, assegna 0

        print("tutte le colonne sono state normalizzate con successo!")
        return df_scalato
class GestisciScaling:
    """
    Gestione dello scaling e salvataggio dei dati.
    """

    @staticmethod
    def scale_features(strategia: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scala il dataset in base alla strategia scelta.

        Args:
            strategia (str): 'normalizza' o 'standardizza'
            data (pd.DataFrame): dataset da scalare

        Returns:
            pd.DataFrame: dataset scalato
        """
        if strategia == 'normalizza':
            return Scalingdf.normalizza(data)
        elif strategia == 'standardizza':
            return Scalingdf.standardizza(data)
        else:
            raise ValueError("Purtroppo non è presente la strategia richiesta, usane una valida")

    @staticmethod
    def gestisci_scaling_e_salva(df: pd.DataFrame, output_path: str) -> None:
        """
        Metodo interattivo che chiede all'utente quale tipo di scaling applicare,
        lo esegue sul dataset e salva il risultato.

        Args:
            df (pd.DataFrame): dataset da scalare
            output_path (str): percorso dove salvare il dataset trasformato
        """
        print("Scegli lo scaling delle features: normalizza o standardizza ")
        strategia = input("Per favore scrivi una sola tecnica: ").strip().lower()

        if strategia not in ['normalizza', 'standardizza']:
            print("Questa tecnica non è disponibile, verrà applicata la normalizzazione di default.")
            strategia = 'normalizza'

        print(f"{strategia.capitalize()}zione in corso...")
        df_scaled = GestisciScaling.scale_features(strategia, df)

        # Salva il dataset scalato
        df_scaled.to_csv(output_path, index=True)
        print(f"Dataset scalato salvato come '{output_path}'")