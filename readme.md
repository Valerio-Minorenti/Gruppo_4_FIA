# AI per la classificazione dei tumori

## Introduzione

Questo progetto mira a sviluppare un programma per analizzare un dataset contenente informazioni su vari tipi di cellule tumorali e determinare se un tumore è benigno o maligno. Il programma utilizza il `kNN_classifier` per effettuare la classificazione dei dati. Le tecniche di valutazione implementate includono `Holdout`, `Random_Subsampling` e `Stratified Cross Validation`, a scelta dell'utente. L'obiettivo finale è testare le prestazioni del modello e valutarne l'efficacia tramite metriche e grafici, che verranno salvati in un file di output. Il programma è stato progettato per essere flessibile e personalizzabile, permettendo all'utente di specificare le opzioni di input.

## Dataset
Dopo una lettura del problema si conclude come il raggiungimento dell’obiettivo finale per classificare i tumori si divida in vari passi. Si è pensato, quindi, di svolgere il primo compito e di eseguire il Data Processing. Per motivi di comodità, si è pensato di voler convertire il file CSV fornito in un file XLSX. Quindi si è deciso di stilare una lista di punti:

1)	Convertire il file;
2)	Eliminare le colonne non utili ai fini del progetto;
3)	Ordinare gli ID;
4)	Eliminare le righe duplicate;
5)	Convertire tutti i valori nelle celle in numeri;
6)	Sostituire i valori numerici errati con valori accettabili;
7)	Correggere i valori sotto la colonna Class Type;
8)	Separare le colonne con i features da quelle con le classi;
9)	Eliminare eventuali righe uguali rimaste.

Le celle con valore pari a 0 sono convertite in 1, e quelle superiore a 10 in valore pari a 10.
Le celle sotto la colonna CLASS TYPE con valore pari a 1 sono convertite in 2.

Il codice è commentato per aiutare l'utente a comprendere il funzionamento di ogni sua parte. Inoltre, mentre viene eseguito, stampa numerosi messaggi nel terminale per indicare cosa sta accadendo in ogni fase dell'elaborazione. 

### Scaling dei Dati
Lo scaling è stato eseguito per rendere le feature comparabili ed evitare che valori con scale diverse influenzino algoritmi basati sulla distanza, come KNN.

Sono state applicate due tecniche:

### Standardizzazione (Z-score normalization)
Trasforma i dati con media 0 e deviazione standard 1.

### Normalizzazione (Min-Max Scaling)
Scala i dati tra 0 e 1.

L’utente sceglie il metodo; in caso di errore viene applicata la normalizzazione di default.

## Stesura del codice

### Installazione dei Requisiti

Prima di eseguire l'applicazione, è fondamentale installare tutte le dipendenze necessarie. Puoi farlo eseguendo il comando `pip install -r requirements.txt` nella directory principale del progetto. Questo comando provvederà a installare tutte le librerie richieste, come numpy, pandas e altre.

### Implementazione del Classificatore k-NN

Questo algoritmo viene utilizzato per classificare i dati del dataset in base alla vicinanza ai punti di dati esistenti. La classe `kNN_classifier` è stata sviluppata per eseguire la classificazione e restituire i risultati in base ai parametri specificati dall'utente.


## Model Evaluation
### Holdout
`Holdout` è una tecnica di validazione che divide il dataset in due set distinti, uno per l'addestramento del modello e uno per la valutazione delle sue prestazioni. La divisione avviene in base alla percentuale scelta dall'utente, ad esempio, 70% per l'addestramento e 30% per il test.

### Implementazione del Random Subsampling

Questa tecnica divide casualmente il dataset in sottoinsiemi di training e test per valutare le prestazioni del modello. La classe `RandomSubsampling` è stata progettata per effettuare questa suddivisione e calcolare le metriche di valutazione.

### Stratified Cross Validation

La Stratified Cross Validation divide il dataset in K- folds mantenendo la proporzione delle classi in ogni fold. Questo assicura che ogni fold sia rappresentativo dell'intero dataset. La classe `StratifiedCrossValidation` è stata sviluppata per eseguire questa tecnica.

### Metriche di valutazione

Le metriche servono per valutare la performance del modello. Si dispone delle seguenti metriche:
- **Accuracy**: La percentuale di classificazioni corrette sul totale delle previsioni.
- **Error Rate**: La percentuale di classificazioni errate sul totale delle previsioni. Questa metrica indica quanto spesso il modello sbaglia nella classificazione dei dati.
- **Sensitivity**: La capacità del modello di identificare correttamente i positivi ( TP/ (TP + FN)).
- **Specificity**: La capacità del modello di identificare correttamente i negativi (TN/ (TN + FP)).
- **Geometric Mean**: La media geometrica di Sensitivity e Specificity, utile per valutare il bilanciamento tra le due metriche.
- **Area Under the Curve**: Una misura della capacità del modello di distinguere tra classi positive e negative.

## Risultati
I risultati dell'esecuzione del programma, a seconda dell'input dell'utente, sono i seguenti:
`Holdout`:
Si ottiene un file di output con il path del file in formato Excel nel quale sono salvate le metriche di validazione del modello e un plot:


- Point Plot: il valore della metrica associata.


`Random Subsampling` e `Stratified Cross Validation`:
Si ottiene un file di output con il path del file in formato Excel, dove sono salvate le metriche di validazione del modello e due plot:


- Line Plot: Illustra per ciascuna metrica come i valori cambiano attraverso diversi esperimenti


- Box Plot: Per ciascuna metrica mostra la distribuzione dei valori.


Facoltativamente, è possibile generare anche:


- Confusion Matrix: Visualizza la matrice di confusione.


- ROC Curve: Visualizza la curva ROC e calcola l'AUC.


### Personalizzazione per l'utente

- Il codice chiede all'utente di specificare i paths di input e output invece di utilizzare percorsi statici predefiniti.
- `numero_di_vicini (k)`: Questo parametro determina il numero di vicini da considerare nell'algoritmo di apprendimento.
- `numero_di_esperimenti (K)`: Questo parametro specifica il numero di esperimenti per eseguire più iterazioni dell'algoritmo, per i metodi `Random Subsampling` e `Stratified Cross Validation`.
- Metodo di valutazione: Puoi scegliere tra Holdout, Random Subsampling e Stratified Cross Validation. Queste opzioni determinano come vengono valutate le performance del modello.
- Path del file di output: Puoi specificare il path del file di output in cui salvare i risultati dell'analisi.