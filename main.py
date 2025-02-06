from Data_Preprocessing.Manipola import elabora_dati
from Data_Preprocessing.Scaling_main import elabora_scaling_e_salvataggio
from Performance_Evaluation.Metrics import MetricsCalculator
from Performance_Evaluation.Visual_Metrics import MetricsSaver
from Validation.validation_main import validation

def main():
    try:
        input_path = input("Inserire il percorso assoluto di input: ").strip().replace('"', '')
        output_path = input("Inserire il percorso assoluto di output: ").strip().replace('"', '')
        elabora_dati(input_path, output_path)

        # SEZIONE: CARICAMENTO E SCALING DEL DATASET
        features, labels, df_scalato = elabora_scaling_e_salvataggio(output_path)
        print("Elaborazione completata con successo.")

        # SEZIONE: Scelta metodo di split / validazione
        dataset_path = input("Inserisci il percorso del dataset scalato (premere invio per usare il file predefinito): ").strip() or "Dati_Progetto_Gruppo4_scalato.csv"

        results = validation(dataset_path, df_scalato)

        if results:
            print("Validazione completata con successo.")

            # SEZIONE: Calcolo e stampa METRICHE
            metrics_calculator = MetricsCalculator()
            metrics_to_calculate = metrics_calculator.scegli_metriche()
            metrics_by_experiment = metrics_calculator.calcola_e_stampa_metriche(results, metrics_to_calculate)

            # SEZIONE: Salvataggio dei risultati (Excel) + Plot confusion matrix e ROC se necessario
            MetricsSaver.salva_risultati_excel(results, metrics_by_experiment)

    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    main()
