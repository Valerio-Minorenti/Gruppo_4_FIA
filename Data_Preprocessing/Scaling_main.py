import pandas as pd
from Data_Preprocessing.Scaling_dati import GestisciScaling


def elabora_scaling_e_salvataggio(output_path):
    # Carica il dataset originale con la colonna 'Sample code number' come indice
    df = pd.read_csv(output_path, index_col="Sample code number")

    # Separa le etichette (classtype_v1) dalle feature
    labels = df["classtype_v1"]
    features = df.drop(columns=["classtype_v1"])

    # Gestisce scaling e salvataggio solo sulle features (senza la colonna 'classtype_v1')
    GestisciScaling.gestisci_scaling_e_salva(features, "Dati_Progetto_Gruppo4_scalato.csv")

    # Ricarica il file salvato con le features scalate
    df_scalato = pd.read_csv("Dati_Progetto_Gruppo4_scalato.csv", index_col="Sample code number")

    # Aggiungi di nuovo la colonna 'classtype_v1' al dataframe scalato
    df_scalato["classtype_v1"] = labels

    # Salva di nuovo il dataframe con la colonna aggiunta
    df_scalato.to_csv("Dati_Progetto_Gruppo4_scalato_con_classtype.csv", index=False)

    # CONVERSIONE A NUMPY
    features = features.to_numpy()
    labels = labels.to_numpy()

    return features, labels,df_scalato # Restituisce features e labels come numpy array
