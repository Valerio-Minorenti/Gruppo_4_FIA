import numpy as np
import pandas as pd
from kNN_classifier import KNN
from Data_Preprocessing.Algoritmo_manipolazione_dati import ManipolaDati
from Validation.Holdout import Holdouts
from Validation.Random_Subsampling import RandomSubsampling
from Validation.Stratified_Cross_Validation import StratifiedCrossValidation
from Data_Preprocessing.Scaling_dati import GestisciScaling
from Performance_Evaluation.Metrics import MetricsCalculator
from Performance_Evaluation.Visual_Metrics import MetricsSaver


def main():
    try:
        # Input e output path che vanno messi dall'utente
        input_path = input("Inserire il percorso assoluto di input: ").strip().replace('"','')
        output_path = input("Inserire il percorso assoluto di output: ").strip().replace('"','')  # Rimuove eventuali virgolette

        # Creazione istanza di classe per manipolare i dati
        manipolatore = ManipolaDati(input_path, output_path)

        # Dopo che sono stati scritti si salvano e printano
        input_path, output_path = manipolatore.salva_percorso()

        # Dopo si carica il file nel df
        df = manipolatore.carica_file(input_path)

        # Passo 1: eliminare le colonne non utili
        colonne_input = input(
            "Scrivi i nomi delle colonne da eliminare separate da una virgola (oppure premi invio se nessuna): ")
        if colonne_input.strip():
            colonne_da_eliminare = [col.strip() for col in colonne_input.split(",")]
        else:
            colonne_da_eliminare = []
        df = manipolatore.elimina_colonne(df, colonne_da_eliminare)

        # Passo 2: ordinare la colonna ID
        df = manipolatore.ordina_colonna(df, "Sample code number")

        # Passo 3: eliminare i duplicati basandosi sugli ID
        df = manipolatore.elimina_duplicati_su_colonna(df, "Sample code number")
        df = df.drop_duplicates()

        # Passo 4: convertire tutti i valori delle celle in numeri
        df = manipolatore.converti_a_numerico(df)

        # Passo 5: Corregge i numeri errati (i >10 o 0 e NaN)
        df = manipolatore.correggi_valori(df)

        # Passo 6: Correggere la colonna class type che ha degli 1 invece dei 2
        df = manipolatore.correggi_class_type(df, ["classtype_v1"])

        # Passo 7: "scambiare" i valori delle colonne così si hanno separati features e type
        df = manipolatore.scambia_colonne(df)

        # Passo ultimo
        manipolatore.salva_file(df, output_path)

        print("Elaborazione completata con successo.")

    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        return # Esci se fallisce la parte di manipolazione dati

    # ------------------------------------------------------------------------------------
    #  SEZIONE: CARICAMENTO E SCALING DEL DATASET
    # ------------------------------------------------------------------------------------
    try:
        # Carica il dataset salvato e usa Sample code number come indice
        df = pd.read_csv(output_path, index_col="Sample code number")

        # Separa le etichette (classtype_v1) dalle feature
        labels = df["classtype_v1"]  # Salva la colonna delle etichette
        features = df.drop(columns=["classtype_v1"])  # Mantieni solo le feature per lo scaling

        # Step 3: Scelta dell'utente per il Feature Scaling
        print("Scegli lo scaling delle features: normalizza o standardizza ")
        strategia = input("Per favore scrivi una sola tecnica: ").strip().lower()

        if strategia not in ['normalizza', 'standardizza']:
            print("Questa tecnica non è disponibile, verrà applicata la normalizzazione di default.")
            strategia = 'normalizza'
        print(f"{strategia}zione in corso...")

        try:
            features_scaled = GestisciScaling.scale_features(strategia=strategia, data=features)
        except Exception as e:
            print(f"C'è stato un errore durante lo scaling ({e}). Usiamo i dati originali.")
            features_scaled = features  # Se fallisce, usa i dati originali

        # Dopo lo scaling, riaggiungi classtype_v1
        df_final = features_scaled.copy()
        df_final["classtype_v1"] = labels

        print("Dati dopo lo scaling delle feature (prime 5 righe):")
        print(df_final.head())

        # CONVERSIONE A NUMPY
        features = features_scaled.to_numpy()
        labels = labels.to_numpy()

    except Exception as e:
        print(f"Si è verificato un errore durante la fase di scaling o caricamento: {e}")
        return

        # Salva il dataset normalizzato/standardizzato (se necessario)
    df_final.to_csv("Dati_Progetto_Gruppo4_scalato.csv", index=True)
    print("Dataset normalizzato/standardizzato salvato come 'Dati_Progetto_Gruppo4_scalato.csv'")
    # ------------------------------------------------------------------------------------
    # SEZIONE: Scelta metodo di split / validazione
    # ------------------------------------------------------------------------------------
    dataset_path = input(
        "Inserisci il percorso del (scalato) da utilizzare (premere invio per usare il file scalato): ").strip()
    if not dataset_path:
        dataset_path = "Dati_Progetto_Gruppo4_scalato.csv"

    # carica dataset
    try:
        df = pd.read_csv(dataset_path)
        print("Dataset letto correttamente.")
    except FileNotFoundError:
        print(f"Errore: Il file '{dataset_path}' non esiste.")
        return

    # chiede quanti vicini per il KNN
    while True:
        try:
            k = int(input("Inserire il numero di vicini per KNN: ").strip())
            if k > 0:
                break
            else:
                print("Errore: Il valore deve essere un numero intero positivo.")
        except ValueError:
            print("Errore: Devi inserire un numero intero positivo.")

    # Presentazione all'utente delle opzioni dei metodi di valutazione e richiesta di una scelta numerica
    print("Scegli il metodo di split desiderato:")
    print("1: Holdout")
    print("2: Random Subsampling")
    print("3: Stratified Cross Validation")

    results = []
    method_choice = None

    while True:
        method_choice = input("Inserisci il numero del metodo di split (1, 2 o 3): ").strip()

        if method_choice == "1":

            try:
                test_ratio = float(input("Inserisci la percentuale dei dati per il testing (es: 0.3 = 30%): ").strip())
                # inizializza Holdout
                holdout = Holdouts(test_ratio=test_ratio, data=features, labels=labels)
                results = holdout.generate_splits(k)  # Passiamo k (numero di vicini)
            except Exception as e:
                print(f"Si è verificato un errore durante l'esecuzione dell'Holdout: {e}")

            break

        elif method_choice == "2":
            try:
                n_folds = int(input("Inserisci il numero di esperimenti (K): "))
                random_subsampling = RandomSubsampling(
                    df=df,
                    x=df.iloc[:, :-1],  # Tutte le colonne tranne l'ultima come feature
                    y=df['classtype_v1'],  # La colonna delle etichette
                    k_experiments=n_folds,  # Numero di esperimenti
                    classifier_class=KNN,  # Classe del classificatore
                    classifier_params={'k': k},  # Parametri del classificatore KNN
                    test_size=0.2  # Percentuale di dati per il test set
                )

                # Genera gli split
                results=random_subsampling.generate_splits()
            except ValueError:
                print("Inserisci un numero intero valido per gli esperimenti e il valore di K > 1.")
            break

        elif method_choice == "3":
            try:
                n_folds = int(input("Inserisci il numero di esperimenti (K): "))
                # Inizializzazione della classe StratifiedCrossValidation
                Strati = StratifiedCrossValidation(
                    K=n_folds,
                    df=df,
                    class_column="classtype_v1",
                    k_neighbors=k
                )

                results=Strati.generate_splits()

            except ValueError:
                print("Inserisci un numero intero valido per gli esperimenti e il valore di K > 1.")

            break

        else:
            print("Scelta non valida. Inserisci '1' (Holdout), '2' (Random), o '3' (Stratified).")

    # ------------------------------------------------------------------------------------
    # SEZIONE: Calcolo e stampa METRICHE
    # ------------------------------------------------------------------------------------
    metrics_calculator = MetricsCalculator()

    # Chiedi le metriche da calcolare
    metrics_to_calculate = metrics_calculator.scegli_metriche()

    # Calcola e stampa le metriche, ottenendo il dizionario con i risultati
    metrics_by_experiment = metrics_calculator.calcola_e_stampa_metriche(results, metrics_to_calculate)

    # ------------------------------------------------------------------------------------
    # SEZIONE: Salvataggio dei risultati (Excel) + Plot confusion matrix e ROC se necessario
    # ------------------------------------------------------------------------------------
    # 1) Chiedi all'utente dove salvare il file Excel
    excel_path = input(
        "\nInserisci il percorso completo per il file Excel dove salvare (es. C:/risultati/risultati_metriche.xlsx): ").strip()
    if not excel_path.lower().endswith(".xlsx"):
        excel_path += ".xlsx"

    # 2) Crea un'istanza di MetricsSaver con il percorso scelto
    visual_metrics = MetricsSaver(filename=excel_path)

    # 3) Salva su Excel tutte le metriche (le liste di K esperimenti per ogni metrica)
    visual_metrics.save_metrics(metrics_by_experiment, sheet_name='Risultati Metriche')

    # 4) Mostra e salva i plot (boxplot e line plot) delle metriche su Excel
    visual_metrics.metrics_plot(metrics_by_experiment)

    # 5) (Facoltativo) Plot Confusion Matrix per ciascun esperimento
    plot_cm = input(
        "\nVuoi plottare e salvare la Confusion Matrix per ognuno dei K esperimenti? (s/n): ").strip().lower()
    if plot_cm == 's':
        for i, (y_test, predictions, predicted_proba) in enumerate(results):
            print(f"Plot Confusion Matrix - Esperimento {i + 1}")
            visual_metrics.plot_confusion_matrix(y_test, predictions)

    # Prepara i risultati per il plot ROC
    new_results = [(y_test, predicted_proba) for y_test, _, predicted_proba in results]

    # 6) (Facoltativo) Plot ROC per ciascun esperimento
    plot_roc = input("\nVuoi plottare e salvare la ROC Curve per ognuno dei K esperimenti? (s/n): ").strip().lower()
    if plot_roc == 's':
        for i, (y_test, predicted_proba) in enumerate(new_results):
            print(f"Plot ROC Curve - Esperimento {i + 1}")
            visual_metrics.plot_roc_curve(y_test, predicted_proba)
    print("\nFine esecuzione. Risultati salvati in:", excel_path)

if __name__ == "__main__":

    main()