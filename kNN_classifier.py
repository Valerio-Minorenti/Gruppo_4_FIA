import random
import numpy as np

class KNN:
    def __init__(self, k):
        """
        Inizializza il classificatore KNN.
        :param k: Numero di vicini da considerare per la classificazione.

        """
        self.k = k
        self.x_train = None # Dati di addestramento (features)
        self.y_train = None # Etichette dei dati di addestramento

    def fit(self, x_train, y_train):
        """
        Memorizza i dati di addestramento.
        :param x_train: Array NumPy contenente le features dei dati di addestramento.
        :param y_train: Array NumPy contenente le etichette dei dati di addestramento.
        """
        self.x_train = x_train
        self.y_train = y_train

    def euclidean_distance(self, x, y):
        """
        Calcola la distanza euclidea tra due punti.
        :param x: Primo punto.
        :param y: Secondo punto.
        :return: La distanza euclidea tra x e y.
        """
        return np.linalg.norm(x - y)

    def nearest_neighbors(self, test_sample):
        """
        Trova i k vicini più vicini al campione di test fornito.
        :param test_sample: Campione di test.
        :return: Indici dei k vicini più vicini nel set di addestramento.
        """
        distances = []  # Lista delle distanze tra il campione di test e i punti di addestramento
        for i in range(len(self.x_train)):
            distances.append(self.euclidean_distance(self.x_train[i], test_sample))  # Calcola la distanza euclidea
        distances = np.argsort(distances)  # Ordina gli indici delle distanze in ordine crescente
        neighbors = distances[:self.k]  # Seleziona i primi k vicini
        return neighbors

    def predict(self, test_train):
        """
        Predice le etichette per i campioni di test.
        """
        predictions = []  # Lista delle etichette predette
        for test_sample in test_train:
            neighbors = self.nearest_neighbors(test_sample)  # Trova i vicini più vicini
            labels = self.y_train[np.array(neighbors)]  # Ottieni le etichette dei vicini
            
            # Conta la frequenza di ogni etichetta tra i vicini
            label_counts = np.bincount(labels.astype(int))
            
            # Trova l'etichetta (o le etichette) con la frequenza massima
            max_count = np.max(label_counts)
            candidates = np.where(label_counts == max_count)[0]
            
            if len(candidates) == 1:
                # Se non c'è pareggio, prendi l'etichetta con il massimo conteggio
                predicted_label = candidates[0]
            else:
                # In caso di pareggio, seleziona un'etichetta casuale tra i candidati
                predicted_label = random.choice(candidates)
            
            predictions.append(predicted_label)  # Aggiungi l'etichetta predetta alla lista delle predizioni
        
        return predictions
