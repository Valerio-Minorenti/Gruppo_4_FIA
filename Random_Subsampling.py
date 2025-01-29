import numpy as np
from kNN_classifier import KNN  # Assicurati che la classe KNN sia definita correttamente

class RandomSubsampling:
    def __init__(self, classifier_class, classifier_params, test_size=0.3):
        """
        Inizializza la classe RandomSubsampling con i parametri specificati.

        :param classifier_class: La classe del classificatore da utilizzare.
        :param classifier_params: Un dizionario contenente i parametri per inizializzare il classificatore.
        :param test_size: La percentuale di dati da utilizzare per il test.
        """
        self.classifier_class = classifier_class
        self.classifier_params = classifier_params
        self.test_size = test_size

    def train_test_split(self, x, y):
        """
        Divide i dati in set di addestramento e di test.

        :param x: Array NumPy contenente le caratteristiche dei dati.
        :param y: Array NumPy contenente le etichette dei dati.
        :return: Tuple contenenti i set di addestramento e di test (x_train, x_test, y_train, y_test).
        """
        indices = np.arange(x.shape[0])  # Crea un array di indici da 0 a n-1, dove n è il numero di campioni
        np.random.shuffle(indices)  # Mescola casualmente gli indici
        split_index = int(x.shape[0] * (1 - self.test_size))  # Calcola l'indice di divisione in base alla dimensione del test set
        train_indices = indices[:split_index]  # Seleziona gli indici per il set di addestramento
        test_indices = indices[split_index:]  # Seleziona gli indici per il set di test
        return x[train_indices], x[test_indices], y[train_indices], y[test_indices]  # Restituisce i set di addestramento e di test

    def run_experiments(self, x, y, k_experiments):
        """
        Esegue il random subsampling per un numero specificato di esperimenti e calcola le accuratezze.

        :param x: Array NumPy contenente le caratteristiche dei dati.
        :param y: Array NumPy contenente le etichette dei dati.
        :param k_experiments: Numero di esperimenti da eseguire.
        :return: Lista delle accuratezze per ciascun esperimento.
        """
        accuracies = []  # Inizializza una lista vuota per memorizzare le accuratezze di ciascun esperimento
        for _ in range(k_experiments):  # Esegue un ciclo per il numero di esperimenti specificato
            x_train, x_test, y_train, y_test = self.train_test_split(x, y)  # Divide i dati in set di addestramento e di test
            classifier = self.classifier_class(**self.classifier_params)  # Crea un'istanza del classificatore con i parametri forniti
            classifier.fit(x_train, y_train)  # Addestra il classificatore sui dati di addestramento
            predictions = classifier.predict(x_test)  # Predice le etichette per i dati di test
            accuracy = np.mean(predictions == y_test)  # Calcola l'accuratezza delle predizioni
            accuracies.append(float(accuracy))  # Aggiunge l'accuratezza alla lista, convertendola in float per evitare np.float64
        return accuracies  # Restituisce la lista delle accuratezze