import pandas as pd
from Holdout import Holdouts  # Importa la classe per Holdout
from Scaling_dati import Scalingdf  # Importa la classe per lo scaling

# percorso csv da mettere, volendo si può mettere direttamente ilnome file
dataset_path = input("Inserisci il percorso assoluto: ").strip()

# chiede come dividere
while True:
    try:
        test_ratio = float(input("inserisci la percentuale dei dati per il testing (0.3 =30%): ").strip())
        if 0 < test_ratio < 1:
            break
        else:
            print("il valore deve essere tra 0 e 1")
    except ValueError:
        print("Errore: devi inserire un numero decimale tra 0 e 1 (es: 0.3 per 30%)!!")

# carica dtaset
try:
    df = pd.read_csv(dataset_path)
    print("datset letto correttamente.")
except FileNotFoundError:
    print(f" Errore: Il file '{dataset_path}'non esiste.")
    exit()

# prende le feature (colonne 2-10) e la label (colonna 11)
features = df.iloc[:, 1:9]  # Colonne da 2 a 10
labels = df.iloc[:, 10]     # Colonna 11

# chiede come scalare i dati
while True:
    scaling_choice = input("Scrivere normalizza o standardizza o nessuna per decidere come scalare i dati: ").strip().lower()
    if scaling_choice in ["normalizza", "standardizza", "nessuna"]:
        break
    else:
        print("Errore: Scegli 'normalizza', 'standardizza' o 'nessuna'.")

# Applica lo scaling solo alle feature (non alla label)
if scaling_choice == "normalizza":
    features = Scalingdf.normalizza(features)
    print("Normalizzazione applicata.")
elif scaling_choice == "standardizza":
    features = Scalingdf.standardizza(features)
    print("Standardizzazione applicata.")
else:
    print("Nessuna trasformazione applicata.")

# converte in modo che sia leggibile in numpy
features = features.to_numpy()
labels = labels.to_numpy()

# chiede quanti vicini per il KNN
while True:
    try:
        k = int(input("inserire il numero di vicini per KNN: ").strip())
        if k > 0:
            break
        else:
            print("Errore: Il valore deve essere un numero intero positivo.")
    except ValueError:
        print("Errore: Devi inserire un numero intero positivo.")

# inizializza Holdout
holdout = Holdouts(test_ratio=test_ratio, data=features, labels=labels)

# inizio test
try:
    results = holdout.generate_splits(k)

    if results:
        y_true, y_pred = results[0]  # estrae etichette reali e previste
        print("Holdout eseguito con successo.")
        print(f"Numero di campioni nel test set: {len(y_true)}")
        print(f"Esempio di etichette reali: {y_true[:10]}")
        print(f"Esempio di etichette predette: {y_pred[:10]}")
    else:
        print("Errore nell'esecuzione dell'Holdout.")
except Exception as e:
    print(f"Errore: {e}")