import numpy as np
import pandas as pd
from kNN_classifier import KNN
from Algoritmo_manipolazione_dati import ManipolaDati
from Holdout import Holdouts
from Random_Subsampling import RandomSubsampling
from Stratified_Cross_Validation import StratifiedCrossValidation
from Scaling_dati import Scalingdf, GestisciScaling
from METRICHE import MetricsCalculator


def main():
    try:
        # Input e output path che vanno messi dall'utente
        input_path = input("Inserire il percorso assoluto di input: ").strip()
        output_path = input("Inserire il percorso assoluto di output: ").strip().replace('"', '')  # Rimuove eventuali virgolette
        
        # Creazione istanza di classe
        manipolatore = ManipolaDati(input_path, output_path)
        
        # Dopo che sono stati scritti si salvano e printano
        input_path, output_path = manipolatore.salva_percorso()

        # Dopo si carica il file nel df
        df = manipolatore.carica_file(input_path)
        
        # Passo 1: eliminare le colonne non utili
        colonne_input = input("Scrivi i nomi delle colonne da eliminare separate da una virgola: ")
        colonne_da_eliminare = [col.strip() for col in colonne_input.split(",")]
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

    # NUOVA SEZIONE: CARICAMENTO E SCALING DEL DATASET
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
            print("Questa tecnica non è disponibile, verrà applicata la normalizzazione ")
            strategia = 'normalizza'
        print(f"{strategia}azione in corso...")

        try:
            features_scaled = GestisciScaling.scale_features(strategia=strategia, data=features)
        except Exception as e:
            print(f"C'è stato un errore {e}, non è stato scalato nulla")
            features_scaled = features  # Se fallisce, usa i dati originali

        # Dopo lo scaling, riaggiungi classtype_v1
        df_final = features_scaled.copy()
        df_final["classtype_v1"] = labels  

        print("Dati dopo lo scaling delle feature:")
        print(df_final.head())

        # CONVERSIONE A NUMPY 
        features = features_scaled.to_numpy()
        labels = labels.to_numpy()

    except Exception as e:
        print(f"Si è verificato un errore: {e}")

    df_final.to_csv("Dati_Progetto_Gruppo4_scalato.csv", index=True)  # Salva il dataset normalizzato
    print(" Dataset normalizzato/standardizzato salvato come 'Dati_Progetto_Gruppo4_scalato.csv'")

    # percorso csv da mettere, volendo si può mettere direttamente il nome file
    dataset_path = input("Inserisci il percorso assoluto: ").strip()

    # carica dataset
    try:
        df = pd.read_csv(dataset_path)
        print("dataset letto correttamente.")
    except FileNotFoundError:
        print(f"Errore: Il file '{dataset_path}' non esiste.")
        exit()

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

    # Presentazione all'utente delle opzioni di split e richiesta di una scelta numerica
    print("Scegli il metodo di split desiderato:")
    print("1: Holdout")
    print("2: Random Subsampling")
    print("3: Stratified Cross Validation")
    
    while True:
        method_choice = input("Inserisci il numero del metodo di split (1, 2 o 3): ").strip()

        if method_choice == "1":
            method_input = "holdout"

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

            # carica dataset
            try:
                df = pd.read_csv(dataset_path)
                print("dataset letto correttamente.")
            except FileNotFoundError:
                print(f"Errore: Il file '{dataset_path}' non esiste.")
                exit()

            # inizializza Holdout
            holdout = Holdouts(test_ratio=test_ratio, data=features, labels=labels)

            # inizio test
            try:
                results = holdout.generate_splits(k)  # type: ignore

                if results:
                    y_true, y_pred = results[0]  # estrae etichette reali e previste
                    print("Holdout eseguito con successo.")
                    print(f"Numero di campioni nel test set: {len(y_true)}")
                    print(f"Esempio di etichette reali: {y_true[:30]}")
                    print(f"Esempio di etichette predette: {[int(pred) for pred in y_pred[:30]]}")
                    # Conteggio dei 4 nelle etichette reali e predette
                    count_4_real = y_true[:].count(4)
                    count_4_pred = [int(pred) for pred in y_pred[:]].count(4)

                    count_2_real = y_true[:].count(2)
                    count_2_pred = [int(pred) for pred in y_pred[:]].count(2)

                    # Stampa dei conteggi
                    print(f"Numero di '4' nelle etichette reali: {count_4_real}")
                    print(f"Numero di '4' nelle etichette predette: {count_4_pred}")
                    print(f"Numero di '2' nelle etichette reali: {count_2_real}")
                    print(f"Numero di '2' nelle etichette predette: {count_2_pred}")
                    
                    # Calcolo delle metriche
                    tp, tn, fp, fn = MetricsCalculator.confu(y_test = y_true,predictions = y_pred)
                    metrics_calc = MetricsCalculator(true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)
                    cm = metrics_calc.confusion_matrix()
                    print("\nMatrice di confusione:")
                    print(cm)

                    metrics = metrics_calc.calculate_metrics(cm)
                    print("\nMetriche calcolate:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")
                else:
                    print("Errore nell'esecuzione dell'Holdout.")
                break  # Esce dal ciclo dopo aver eseguito l'Holdout
            except Exception as e:
                print(f"Si è verificato un errore durante l'esecuzione dell'Holdout: {e}")

        elif method_choice == "2":
            method_input = "Random Subsampling"

            while True:
                n_folds_str = input("Inserisci il numero di esperimenti K (ad esempio, 5): ")
                try:
                    n_folds = int(n_folds_str)
                    if n_folds > 1:
                        break
                    else:
                        raise ValueError
                except ValueError:
                    print("Il numero di esperimenti K deve essere un numero intero positivo.")
            try:
                df = pd.read_csv(dataset_path)
                print("Dataset letto correttamente.")
        
            except FileNotFoundError:
                print(f"Errore: Il file '{dataset_path}' non esiste.")
                exit()


            # Inizializzazione di Random Subsampling
            try:
                random_subsampling = RandomSubsampling(
                    df = df,
                    x=df.iloc[:, :-1],        # Assumendo che la prima colonna sia un ID o indice
                    y=df['classtype_v1'],      # La colonna delle etichette
                    k_experiments=n_folds,     # Numero di esperimenti
                classifier_class=KNN,      # Classe del classificatore
                    classifier_params={'k': k},  # Parametri del classificatore
                    test_size=0.2              # Percentuale di dati per il test set
                )
            except Exception as e:
                print(f"Errore durante l'inizializzazione del Random Subsampling: {e}")
                exit()

                # Esegui gli esperimenti
            try:
                results = random_subsampling.run_experiments()
                if not results:
                    raise ValueError("Nessun risultato è stato generato. Verifica i dati di input e i parametri.")

    # Stampa dei risultati degli esperimenti
                for experiment_index, (y_test, predictions) in enumerate(results):
                    print(f"\nEsperimento {experiment_index + 1}:")

            # Stampa delle etichette reali e delle predizioni
                    print("\nEtichette reali (y_test):", y_test)
                    print("Predizioni (predictions):", predictions)

        # Calcolo delle metriche
                    tp, tn, fp, fn = MetricsCalculator.confu(y_test, predictions)
                    metrics_calc = MetricsCalculator(true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)
                    cm = metrics_calc.confusion_matrix()
                    print("\nMatrice di confusione:")
                    print(cm)

                    metrics = metrics_calc.calculate_metrics(cm)
                    print("\nMetriche calcolate:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")

            except ValueError as ve:
                print(f"Errore nei dati di input o nei parametri: {ve}")
            except Exception as e:
                print(f"Si è verificato un errore durante l'esecuzione del Random Subsampling: {e}")

            break
    
        elif method_choice == "3":
            method_input = "stratified cross validation"

            while True:
                n_folds_str = input("Inserisci il numero di esperimenti K (ad esempio, 5): ")
                try:
                    n_folds = int(n_folds_str)
                    if n_folds > 1:
                        break
                    else:
                        raise ValueError
                except ValueError:
                    print("Il numero di esperimenti K deve essere un numero intero positivo.")

            # Carica dataset
            try:
                df = pd.read_csv(dataset_path)
                print("Dataset letto correttamente.")
            except FileNotFoundError:
                print(f"Errore: Il file '{dataset_path}' non esiste.")
                exit()

            # Istanziazione della classe StratifiedCrossValidation
            Start = StratifiedCrossValidation(K=n_folds, df=df, class_column="classtype_v1", k_neighbors=k)

            # Eseguiamo gli esperimenti stratificati
            results = Start.run_experiments()

            try:
                if not results:
                    raise ValueError("Nessun risultato è stato generato. Verifica i dati di input e i parametri.")

                # Stampa dei risultati degli esperimenti
                for experiment_index, (y_test, predictions) in enumerate(results):
                    print(f"\nEsperimento {experiment_index + 1}:")

                    # Conteggio e stampa delle distribuzioni delle classi nel test set
                    count_4_Test = list(y_test).count(4)
                    count_2_Test = list(y_test).count(2)

                    print(f"Numero di '4' nel test set: {count_4_Test}")
                    print(f"Numero di '2' nel test set: {count_2_Test}")

                    if count_4_Test + count_2_Test > 0:
                        print(f"Percentuale di 4 nel test: {count_4_Test / (count_4_Test + count_2_Test):.2%}")
                        print(f"Percentuale di 2 nel test: {count_2_Test / (count_4_Test + count_2_Test):.2%}")

                        # Stampa delle etichette reali e delle predizioni
                        print("\nEtichette reali (y_test):", [int(val) for val in y_test])
                        print("Predizioni (predictions):", [int(val) for val in predictions])

                    # Calcolo delle metriche
                    tp, tn, fp, fn = MetricsCalculator.confu(y_test, predictions)
                    metrics_calc = MetricsCalculator(true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)
                    cm = metrics_calc.confusion_matrix()
                    print("\nMatrice di confusione:")
                    print(cm)

                    metrics = metrics_calc.calculate_metrics(cm)
                    print("\nMetriche calcolate:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")

            except ValueError as ve:
                print(f"Errore nei dati di input o nei parametri: {ve}")

            break
            
        else:
            print("Scelta non valida. Inserisci '1' per Holdout, '2' per Random Subsampling o '3' per Stratified Cross Validation.")
    

if __name__ == "__main__":
    main()