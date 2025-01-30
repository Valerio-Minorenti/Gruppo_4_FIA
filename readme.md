# AI per la classificazione dei tumori

## Introduzione

Questo progetto mira a sviluppare un programma per analizzare un dataset contenente informazioni su vari tipi di cellule tumorali e determinare se un tumore è benigno o maligno. Il programma utilizza il `kNN_classifier` per effettuare la classificazione dei dati. Le tecniche di valutazione implementate includono `Holdout`, `Random_Subsampling` e `Stratified Cross Validation`, a scelta dell'utente. L'obiettivo finale è testare le prestazioni del modello e valutarne l'efficacia tramite metriche e grafici, che verranno salvati in un file di output. Il programma è stato progettato per essere flessibile e personalizzabile, permettendo all'utente di specificare le opzioni di input.

## Dataset

## Stesura del codice

### Installazione dei Requisiti

Prima di eseguire l'applicazione, è fondamentale installare tutte le dipendenze necessarie. Puoi farlo eseguendo il comando `pip install -r requirements.txt` nella directory principale del progetto. Questo comando provvederà a installare tutte le librerie richieste, come numpy, pandas e altre.

### Implementazione del Classificatore k-NN

Questo algoritmo viene utilizzato per classificare i dati del dataset in base alla vicinanza ai punti di dati esistenti. La classe `kNN_classifier` è stata sviluppata per eseguire la classificazione e restituire i risultati in base ai parametri specificati dall'utente.


## Tecniche di Valutazione

### Implementazione del Random Subsampling

Questa tecnica divide casualmente il dataset in sottoinsiemi di training e test per valutare le prestazioni del modello. La classe `RandomSubsampling` è stata progettata per effettuare questa suddivisione e calcolare le metriche di valutazione.

### Stratified Cross Validation

La Stratified Cross Validation divide il dataset in K- folds mantenendo la proporzione delle classi in ogni fold. Questo assicura che ogni fold sia rappresentativo dell'intero dataset. La classe `StratifiedCrossValidation` è stata sviluppata per eseguire questa tecnica.

## Metriche di valutazione

Le metriche servono per valutare la performance del modello. Si dispone delle seguenti metriche:
- **Accuracy**: La percentuale di classificazioni corrette sul totale delle previsioni.
- **Error Rate**: La percentuale di classificazioni errate sul totale delle previsioni. Questa metrica indica quanto spesso il modello sbaglia nella classificazione dei dati.
- **Sensitivity**: La capacità del modello di identificare correttamente i positivi ( TP/ (TP + FN)).
- **Specificity**: La capacità del modello di identificare correttamente i negativi (TN/ (TN + FP)).
- **Geometric Mean**: La media geometrica di Sensitivity e Specificity, utile per valutare il bilanciamento tra le due metriche. (inoltre nella file `metrics.py` è stata calcolata la media geometrica senza effettuare la radice quadrata poichè va ad essere computazionalmente meno oneroso ed il significato rimane inalterato, poichè più è grande la media geometrica e migliore è il modello).
- **Area Under the Curve**: Una misura della capacità del modello di distinguere tra classi positive e negative.

### Personalizzazione per l'utente

- `numero_di_vicini (k)`: Questo parametro determina il numero di vicini da considerare nell'algoritmo di apprendimento.
- `numero_di_esperimenti (K)`: Questo parametro specifica il numero di esperimenti per eseguire più iterazioni dell'algoritmo, per i metodi `Random Subsampling` e `Stratified Cross Validation`.

