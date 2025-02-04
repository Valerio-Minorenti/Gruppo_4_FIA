from abc import ABC, abstractmethod


class ValidationStrategy(ABC):
    """
    Classe astratta che definisce l'interfaccia per diverse strategie di validazione.
    """

    @abstractmethod
    def generate_splits(self, k=None):
        """
        Metodo astratto che deve essere implementato dalle sottoclassi.
        Deve restituire una lista di tuple (y_test, predictions).

        :param k: Parametro opzionale; nelle sottoclassi può indicare il numero di esperimenti o di fold.
        """
        pass