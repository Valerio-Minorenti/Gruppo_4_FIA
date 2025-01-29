# AI per la classificazione dei tumori

## Introduzione
 Questo progetto mira a sviluppare un programma per analizzare un dataset contenente informazioni su vari tipi di cellule tumorali e determinare se un tumore è benigno o maligno. Il programma utilizza il `kNN_classifier` per effettuare la classificazione dei dati. Le tecniche di valutazione implementate includono `Holdout`, `Random_Subsampling` e `Stratified Cross Validation`, a scelta dell'utente. L'obiettivo finale è testare le prestazioni del modello e valutarne l'efficacia tramite metriche e grafici, che verranno salvati in un file di output. Il programma è stato progettato per essere flessibile e personalizzabile, permettendo all'utente di specificare le opzioni di input.

 ## Dataset

 ## Stesura del codice

 - **Implementazione del Classificatore k-NN**
 Questo algoritmo viene utilizzato per classificare i dati del dataset in base alla vicinanza ai punti di dati esistenti. La classe `kNN_classifier `è stata sviluppata per eseguire la classificazione e restituire i risultati in base ai parametri specificati dall'utente.

 - **Implementazione del Random Subsampling**
 Questa tecnica divide casualmente il dataset in sottoinsiemi di training e test per valutare le prestazioni del modello.  La classe `RandomSubsampling` è stata progettata per effettuare questa suddivisione e calcolare le metriche di valutazione.


