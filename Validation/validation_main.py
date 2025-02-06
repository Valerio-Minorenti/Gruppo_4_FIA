import pandas as pd
from Validation.Holdout import Holdouts
from Validation.Random_Subsampling import RandomSubsampling
from Validation.Stratified_Cross_Validation import StratifiedCrossValidation
from kNN_classifier import KNN


def validation(dataset_path, df_scalato):
    # Carica dataset
    try:

        df = pd.read_csv(dataset_path)
        print("Dataset letto correttamente.")
    except FileNotFoundError:
        print(f"Errore: Il file '{dataset_path}' non esiste.")
        return None  # Esce dalla funzione

    # Chiede quanti vicini per il KNN
    while True:
        try:
            k = int(input("Inserire il numero di vicini per KNN: ").strip())
            if k > 0:
                break
            else:
                print("Errore: Il valore deve essere un numero intero positivo.")
        except ValueError:
            print("Errore: Devi inserire un numero intero positivo.")

    # Presenta le opzioni dei metodi di valutazione
    print("Scegli il metodo di split desiderato:")
    print("1: Holdout")
    print("2: Random Subsampling")
    print("3: Stratified Cross Validation")

    results = []

    while True:
        method_choice = input("Inserisci il numero del metodo di split (1, 2 o 3): ").strip()

        if method_choice == "1":
            try:
                test_ratio = float(input("Inserisci la percentuale dei dati per il testing (es: 0.3 = 30%): ").strip())

                # Modifica: ora passiamo solo `data` (che contiene sia le feature che la colonna `classtype_v1`)
                holdout = Holdouts(test_ratio=test_ratio, data=df_scalato)

                # Passiamo k (numero di vicini) per generare i risultati
                results = holdout.generate_splits(k)

            except Exception as e:
                print(f"Si è verificato un errore durante l'esecuzione dell'Holdout: {e}")
            break

        elif method_choice == "2":
            try:
                n_folds = int(input("Inserisci il numero di esperimenti (K): "))
                random_subsampling = RandomSubsampling(
                    df=df_scalato,
                    x=df_scalato.iloc[:, :-1],  # Tutte le colonne tranne l'ultima come feature
                    y=df_scalato['classtype_v1'],  # La colonna delle etichette
                    k_experiments=n_folds,  # Numero di esperimenti
                    classifier_class=KNN,  # Classe del classificatore
                    classifier_params={'k': k},  # Parametri del classificatore KNN
                    test_size=0.2  # Percentuale di dati per il test set
                )
                results = random_subsampling.generate_splits()
            except ValueError:
                print("Inserisci un numero intero valido per gli esperimenti e il valore di K > 1.")
            break

        elif method_choice == "3":
            try:
                n_folds = int(input("Inserisci il numero di esperimenti (K): "))
                stratified_cv = StratifiedCrossValidation(
                    K=n_folds,
                    df=df_scalato,
                    class_column="classtype_v1",
                    k_neighbors=k
                )
                results = stratified_cv.generate_splits()
            except ValueError:
                print("Inserisci un numero intero valido per gli esperimenti e il valore di K > 1.")
            break

        else:
            print("Scelta non valida. Inserisci '1' (Holdout), '2' (Random), o '3' (Stratified).")

    return results  # Restituisce i risultati della validazione
