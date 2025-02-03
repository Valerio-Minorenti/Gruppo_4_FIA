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
        input_path = input("Inserire il percorso assoluto di input (CSV): ").strip()
        output_path = input("Inserire il percorso assoluto di output (CSV pulito): ").strip().replace('"', '')  # Rimuove eventuali virgolette
        
        # Creazione istanza di classe per manipolare i dati
        manipolatore = ManipolaDati(input_path, output_path)
        
        # Dopo che sono stati scritti si salvano e printano
        input_path, output_path = manipolatore.salva_percorso()

        # Dopo si carica il file nel df
        df = manipolatore.carica_file(input_path)
        
        # Passo 1: eliminare le colonne non utili
        colonne_input = input("Scrivi i nomi delle colonne da eliminare separate da una virgola (oppure premi invio se nessuna): ")
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
        return  # Esci se fallisce la parte di manipolazione dati

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
    dataset_path = input("Inserisci il percorso del CSV (scalato) da utilizzare (o premi Invio per usare 'Dati_Progetto_Gruppo4_scalato.csv'): ").strip()
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

    # Presentazione all'utente delle opzioni di split e richiesta di una scelta numerica
    print("Scegli il metodo di split desiderato:")
    print("1: Holdout")
    print("2: Random Subsampling")
    print("3: Stratified Cross Validation")
    
    results = []
    method_choice = None

    while True:
        method_choice = input("Inserisci il numero del metodo di split (1, 2 o 3): ").strip()

        if method_choice == "1":
            method_input = "holdout"

            # chiede come dividere
            while True:
                try:
                    test_ratio = float(input("Inserisci la percentuale dei dati per il testing (es: 0.3 = 30%): ").strip())
                    if 0 < test_ratio < 1:
                        break
                    else:
                        print("Il valore deve essere tra 0 e 1 (es: 0.3).")
                except ValueError:
                    print("Errore: devi inserire un numero decimale (es: 0.3 per 30%).")

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
                    count_4_real = y_true.count(4)
                    count_4_pred = [int(pred) for pred in y_pred].count(4)

                    count_2_real = y_true.count(2)
                    count_2_pred = [int(pred) for pred in y_pred].count(2)

                    # Stampa dei conteggi
                    print(f"Numero di '4' nelle etichette reali: {count_4_real}")
                    print(f"Numero di '4' nelle etichette predette: {count_4_pred}")
                    print(f"Numero di '2' nelle etichette reali: {count_2_real}")
                    print(f"Numero di '2' nelle etichette predette: {count_2_pred}")
                    
            except Exception as e:
                print(f"Si è verificato un errore durante l'esecuzione dell'Holdout: {e}")
            
            break

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
                    print("Il numero di esperimenti K deve essere un numero intero > 1.")

            # Inizializzazione di Random Subsampling
            try:
                random_subsampling = RandomSubsampling(
                    df=df,
                    x=df.iloc[:, :-1],          # Tutte le colonne tranne l'ultima come feature
                    y=df['classtype_v1'],       # La colonna delle etichette
                    k_experiments=n_folds,      # Numero di esperimenti
                    classifier_class=KNN,       # Classe del classificatore
                    classifier_params={'k': k}, # Parametri del classificatore KNN
                    test_size=0.2               # Percentuale di dati per il test set
                )
            except Exception as e:
                print(f"Errore durante l'inizializzazione del Random Subsampling: {e}")
                return

            # Esegui gli esperimenti
            try:
                results = random_subsampling.run_experiments()
                if not results:
                    raise ValueError("Nessun risultato è stato generato. Verifica i dati di input e i parametri.")

                # Stampa dei risultati degli esperimenti (facoltativa)
                for experiment_index, (y_test, predictions) in enumerate(results):
                    print(f"\nEsperimento {experiment_index + 1}:")
                    print("Etichette reali (y_test):", y_test[:10], "...")
                    print("Predizioni (predictions):", predictions[:10], "...")
            except ValueError as ve:
                print(f"Errore nei dati di input o nei parametri: {ve}")
            except Exception as e:
                print(f"Si è verificato un errore durante l'esecuzione del Random Subsampling: {e}")

            break

        elif method_choice == "3":
            method_input = "Stratified Cross Validation"

            while True:
                n_folds_str = input("Inserisci il numero di esperimenti K (ad esempio, 5): ")
                try:
                    n_folds = int(n_folds_str)
                    if n_folds > 1:
                        break
                    else:
                        raise ValueError
                except ValueError:
                    print("Il numero di esperimenti K deve essere un numero intero > 1.")

            # Istanziazione della classe StratifiedCrossValidation
            Start = StratifiedCrossValidation(
                K=n_folds, 
                df=df, 
                class_column="classtype_v1", 
                k_neighbors=k
            )

            # Eseguiamo gli esperimenti stratificati
            try:
                results = Start.run_experiments()
                if not results:
                    raise ValueError("Nessun risultato generato. Controlla i dati di input/parametri.")

                for experiment_index, (y_test, predictions) in enumerate(results):
                    print(f"\nEsperimento {experiment_index + 1}:")
                    count_4_test = list(y_test).count(4)
                    count_2_test = list(y_test).count(2)

                    print(f"Numero di '4' nel test set: {count_4_test}")
                    print(f"Numero di '2' nel test set: {count_2_test}")

                    if (count_4_test + count_2_test) > 0:
                        print(f"Percentuale di 4: {count_4_test / (count_4_test + count_2_test):.2%}")
                        print(f"Percentuale di 2: {count_2_test / (count_4_test + count_2_test):.2%}")

                    print("\nEsempio prime 10 etichette reali (y_test):", list(y_test)[:10])
                    print("Esempio prime 10 predizioni (predictions):", list(predictions)[:10])

            except ValueError as ve:
                print(f"Errore nei dati di input o parametri: {ve}")
            except Exception as e:
                print(f"Si è verificato un errore durante la Stratified Cross Validation: {e}")

            break

        else:
            print("Scelta non valida. Inserisci '1' (Holdout), '2' (Random), o '3' (Stratified).")

    # ------------------------------------------------------------------------------------
    # SEZIONE: Calcolo e stampa METRICHE
    # ------------------------------------------------------------------------------------
    # Chiedi all'utente quali metriche calcolare
    metrics_to_calculate = MetricsCalculator.scegli_metriche()

    # Inizializza un dizionario per raccogliere i valori delle metriche per K esperimenti
    metrics_by_experiment = {metric: [] for metric in metrics_to_calculate}

    try:
        # Ciclo su tutti gli esperimenti eseguiti (results è una lista di tuple (y_test, y_pred))
        for experiment_index, (y_test, predictions) in enumerate(results):
            print(f"\nEsperimento {experiment_index + 1} - Calcolo metriche:")

            # Calcolo delle metriche
            tp, tn, fp, fn = MetricsCalculator.confu(y_test, predictions)
            metrics_calc = MetricsCalculator(true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)
            cm = metrics_calc.confusion_matrix()
            print("\nMatrice di confusione:")
            print(cm)

            # Calcola le metriche richieste
            metrics = metrics_calc.calculate_metrics(cm, metrics_to_calculate)

            # Aggiungi ogni metrica all'array corrispondente (per grafici cumulativi)
            for metric, value in metrics.items():
                if metric in metrics_by_experiment:
                    metrics_by_experiment[metric].append(value)

            # Stampa delle metriche per questo esperimento
            print("\nMetriche calcolate (singolo esperimento):")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    except ValueError as ve:
        print(f"Errore nei dati di input o nei parametri: {ve}")
    except Exception as e:
        print(f"Si è verificato un errore durante il calcolo delle metriche: {e}")
        return

    # Calcolo della media delle metriche su K esperimenti
    avg_metrics = {metric: np.mean(values) for metric, values in metrics_by_experiment.items()}
    print("\nMedia delle metriche su tutti gli esperimenti:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Stampa dei risultati delle metriche (lista per ogni esperimento)
    print("\nValori delle metriche per tutti gli esperimenti:")
    for metric, values in metrics_by_experiment.items():
        print(f"{metric}: {values}")

    # ------------------------------------------------------------------------------------
    # SEZIONE: Salvataggio dei risultati (Excel) + Plot confusion matrix e ROC se necessario
    # ------------------------------------------------------------------------------------
    # 1) Chiedi all'utente dove salvare il file Excel
    excel_path = input("\nInserisci il percorso completo per il file Excel dove salvare (es. C:/risultati/risultati_metriche.xlsx): ").strip()
    if not excel_path.lower().endswith(".xlsx"):
        excel_path += ".xlsx"

    # 2) Crea un'istanza di MetricsSaver con il percorso scelto
    visual_metrics = MetricsSaver(filename=excel_path)

    # 3) Salva su Excel tutte le metriche (le liste di K esperimenti per ogni metrica)
    visual_metrics.save_metrics(metrics_by_experiment, sheet_name='Risultati Metriche')

    # 4) Mostra e salva i plot (boxplot e line plot) delle metriche su Excel
    visual_metrics.metrics_plot(metrics_by_experiment)

    # 5) (Facoltativo) Plot Confusion Matrix per ciascun esperimento
    plot_cm = input("\nVuoi plottare e salvare la Confusion Matrix per ognuno dei K esperimenti? (s/n): ").strip().lower()
    if plot_cm == 's':
        for i, (y_test, predictions) in enumerate(results):
            print(f"Plot Confusion Matrix - Esperimento {i+1}")
            # Per non sovrascrivere lo stesso foglio, passiamo un nome diverso
            # (se la tua classe non supporta parametri aggiuntivi, useremo lo standard "Confusion Matrix")
            # Oppure possiamo salvare la CM in un unico foglio con più immagini una sotto l'altra.
            # Esempio: togliamo la rotazione per salvare le immagini in posizioni diverse (advanced).
            # Qui semplifichiamo e usiamo la stessa funzione che salva in "Confusion Matrix".
            # NOTA: Rischi di sovrascrivere l'immagine. Se vuoi fogli diversi:
            #   visual_metrics.plot_confusion_matrix(y_test, predictions, sheet_name=f"CM_{i+1}")
            visual_metrics.plot_confusion_matrix(y_test, predictions)

    # 6) (Facoltativo) Plot ROC per ciascun esperimento
    #    Per fare la ROC abbiamo bisogno di punteggi (y_score). Se non li abbiamo, potremmo saltare.
    #    In questo esempio, KNN di solito non li fornisce. Se non li hai, non puoi plottare la ROC in modo corretto.
    #    Se vuoi tentare comunque, ti chiedo se hai i punteggi in results:
    plot_roc = input("\nVuoi plottare e salvare la ROC Curve per ognuno dei K esperimenti? (s/n): ").strip().lower()
    if plot_roc == 's':
        # Qui servirebbe y_score, che la tua attuale implementazione KNN potrebbe non restituire.
        # Ti mostro come farlo, assumendo che "predictions" contenga qualche punteggio.
        # In realtà, se "predictions" è solo etichette discrete, la ROC non ha senso.
        # Se vuoi usare un "fake" punteggio, potresti per esempio mappare 2->0.2 e 4->0.8, ma non è una vera ROC.
        # Adatterai tu, in base al codice del kNN.

        for i, (y_test, predictions) in enumerate(results):
            # Esempio: costruiamo un fake y_score (0.5 se predice 4, 0.1 se predice 2).
            # NON è un vero punteggio, serve solo a mostrare come fare.
            y_score_fake = []
            for pred in predictions:
                if pred == 4:
                    y_score_fake.append(0.9)
                else:
                    y_score_fake.append(0.1)
            print(f"Plot ROC Curve - Esperimento {i+1}")
            visual_metrics.plot_roc_curve(y_test, y_score_fake)

    print("\nFine esecuzione. Risultati salvati in:", excel_path)

if __name__ == "__main__":
    main()
