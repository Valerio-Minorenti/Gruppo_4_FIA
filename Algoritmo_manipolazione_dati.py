import pandas as pd # type: ignore
import os

class Manipola_Dati:
#PASSO 0
    def __init__(self, input_path: str, output_path: str):
        """
        Inizializza la classe con i percorsi di input e output.
        """
        self.input_path = input_path
        self.output_path = output_path



    def salva_percorso(self):
        """ 
        Stampa e restituisce i percorsi di input e output. 
        """
        
        print(f"Percorso del file di input: {self.input_path}")
        print(f"Percorso del file di output: {self.output_path}")
        return self.input_path, self.output_path


#Passo 0.5
    def carica_file(self):
     """
     il file tabulare lo si carica come df

     Args:
         self.input_path (str): percorso dell'input

     Returns:
        pd.DataFrame: df con i dati
     """
     ext = os.path.splitext(self.input_path)[1].lower()  # estrae l'estensione del file
    
     try:
        if ext == ".csv":
            df = pd.read_csv(self.input_path)
        elif ext == ".xlsx":
            df = pd.read_excel(self.input_path)
        elif ext == ".tsv":
            df = pd.read_csv(self.input_path, sep='\t')
        else:
            print(f"Errore: l'estensione {ext} non è supportata, aggiungila.")
            return None
        print("file caricato correttamente")
        return df
     except FileNotFoundError:
        print(f"Errore! {self.input_path} non esiste o è errato il path")
        return None
     except Exception as e:
        print(f"errore nel caricamento {e}")
        return None
    
#PAsso 1
    @staticmethod
    def elimina_colonne(df, colonne_da_eliminare):
     """
    specifichi le colonne nel dataframe  e vengono eliminate, solo nomi di colonne non indici

    args:
        df (pd.DataFrame): il df da cui si eliminano le colonne.
        colonne_da_eliminare (list): la lista dove ci sono le colonne da eliminare.

    Returns:
        pd.DataFrame: lo stesso df ma senza le colonne che sono state scritte.
    """
     df = df.drop(columns=colonne_da_eliminare, errors='ignore')
     print(f"le colonne eliminate sono: {colonne_da_eliminare}")
     return df


#Passo 2
    @staticmethod
    def ordina_colonna(df, colonna_id):
     """
    ordina una colonna specifica in ordine crescente

    Args:
        df (pd.DataFrame): il df da ordinare.
        colonna_id (str): Il nome della colonna da ordinare.

    Returns:
        pd.DataFrame: df con la colonna scritta però ordinata in modo crescente
    """
     if colonna_id in df.columns:
        df = df.sort_values(by=colonna_id)
        print(f"la colonna {colonna_id} è stata ordinata in modo crescente")
     else:
        print(f"non esiste alcuna colonna intitolata {colonna_id}, nessuna colonna è stat ordinata")
     return df


#PAsso 3
    @staticmethod
    def elimina_duplicati_su_colonna(df, colonna):
     """
    elimina i duplicati(righe) prendnedo come riferimento una colonna
    elimina anche le righe che per quella colonna hanno valori vuoti o NaN
    

    Args:
        df (pd.DataFrame): il df da ordinare.
        colonna (str): nome colonna su cui basarsi.

    Returns:
        pd.DataFrame: il df ripulito dai valori duplicati  o nulli
    """
     if colonna in df.columns:
        # Elimina le righe con valori nulli o vuoti nella colonna selezionata
        df = df.dropna(subset=[colonna])  # Elimina righe con NaN nella colonna
        df = df[df[colonna] != ""]  # Elimina righe con stringa vuota nella colonna
        
        # viene mantenuta solo la prima colonna "duplicata"
        df = df.drop_duplicates(subset=[colonna])
        
        print(f"eliminate le righe associate a tutti i duplciati e i valori nulli della colonna {colonna}.")
     else:
        print(f"Attenzione! Non esiste alcuna colonna chiamata {colonna}.")

     return df


#PASSO 4
    @staticmethod
    def converti_a_numerico(df):
     """
    tutte le celle di tutte le colonna diventano valori accettabili e numeri interi

    Args:
        df (pd.DataFrame): df da manipolare

    Returns:
        pd.DataFrame: df con le colonne di soli numeri.
    """
     for col in df.columns:
        # Sostituisco le virgole con i puntile virgole diventano punti
        df[col] = df[col].replace({',': '.'}, regex=True)
        
        # converto tutto in numeri
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Usa 'coerce' per evitare errori di conversione

        # serve per arrotondare 
        df[col] = df[col].round(0).astype("Int64")  # Int64 gestisce anche i valori NaN, che non dovrebbero esserci
        
     print("Tutte le colonne ora sono formate da valori numerici")
     return df



#PASSO 5
    @staticmethod
    def correggi_valori(df):
     """
   Corregge(sostituisce) i valori numerici nelle colonne che non sono quella di ID
   in questo caso:
       1- se il numero è 0 o NaN diventa 1
       2- se è >10 ma divisibile per 10 senza resto diventa quel numero
       3- se è >10 ma non divisibile intero per 10 diventa 9
   
    Args:
        df (pd.DataFrame): Il df da correggere.

    Returns:
        pd.DataFrame: Il df con i valori sostituii
    """
     for col in df.columns[1:]:  # parte da  dopo la prima colonna 
        df[col] = df[col].apply(lambda x: (
            1 if pd.isna(x) or x == 0 else  # NaN o 0 = 1
            int(x // 10) if x > 10 and x % 10 == 0 else  # se >10 e %10=0 diventa il numero
            9 if x > 10 else  # se >10 e non 2, duventa 9
            int(x)  # altri casi son o tutti uguali
        ))

     return df


#Passo 6
    @staticmethod
    def correggi_class_type(df, colonne_da_correggere):
     """
    corregge tutti i valori 1 in 2, serve per la classtype_v1.

    Args:
        df (pd.DataFrame): Il df da correggere.
        colonne_da_correggere (list): colonna da correggere.

    Returns:
        pd.DataFrame: Il df con le colonne corrette.
    """
     for col in colonne_da_correggere:
        if col in df.columns:
            df[col] = df[col].replace(1, 2)  # gli 1 diventano 2 
            print(f"Gli 1 nella colonna '{col}' sono stati sostituiti con 2.")
        else:
            print(f"Attenzione! La colonna '{col}' non esiste .")

     return df

#Passo 7
    @staticmethod
    def scambia_colonne(df, colonna1="classtype_v1", colonna2="bareNucleix_wrong"):
     """
    "Scambia" i valori e i nomi di due colonne.

    Args:
        df (pd.DataFrame): Il df su cui effettuare lo scambio.
        colonna1 (str):   prima colonna.
        colonna2 (str):  seconda colonna.

    Returns:
        pd.DataFrame: Il df con le colonne scambiate.
    """
     if colonna1 in df.columns and colonna2 in df.columns:
        # imverte i valori delle colonne
        df[colonna1], df[colonna2] = df[colonna2], df[colonna1]

        # i nomi si scambiano
        df = df.rename(columns={colonna1: colonna2, colonna2: colonna1})

        print(f"{colonna1} e {colonna2} sono state invertite.")
     else:
        print(f"Attenzione! {colonna1} o {colonna2} non esistono.")

     return df

#PAsso ultimo
    def salva_file(self, df):
     """
    salva il df nel formato tabulare desiderato (supportato)

    Args:
        df (pd.DataFrame): Il df da salvare.
        output_path (str): path del file di output.

    Returns:
        None
     """
     ext = os.path.splitext(self.output_path)[1].lower()  # estrae l'estensione
    
     try:
        if ext == ".csv":
            df.to_csv(self.output_path, index=False)
        elif ext == ".xlsx":
            df.to_excel(self.output_path, index=False, engine='openpyxl')  # engine
        elif ext == ".tsv":
            df.to_csv(self.output_path, sep='\t', index=False)
        else:
            print(f"attenzione! l'estensione {ext} non è supportata, aggiungila")
            return
        print(f"Fìile salvato con successo in: {self.output_path}")
     except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")
        
        
        #####################
        """
#PASSO 0: l'utente deve avere due file tabulari e deve specificare i percorsi
 ##Da qui si chiamano le funzione e l'utente deve scrivere
if __name__ == "__main__":
    # scrivere i percorsi qui, se il  file tabulare è diverso da csv,tsv o xlsx va aggiunto nella funzione salvafile!
    input_path = input("Inserisci il percorso completo del file di input: ").strip()
    output_path = input("Inserisci il percorso completo del file di output: ").strip()
 

   #  sono salvati e vengono anche ridati
    input_path, output_path = salva_percorso(input_path, output_path)

    # Carica il file nel df
    df = carica_file(input_path)

   # PASSO 1: eliminare le colonne non utili ai fini del progetto
   # l'utente deve scrivere i nomi delle colonne. vanno rispettati i caretteri e più colonne devono essere separate da virgole(scrivere nell'input?) 
    colonne_input = input("Inserisci i nomi delle colonne da eliminare, separati da una virgola: ") 
    colonne_da_eliminare = [col.strip() for col in colonne_input.split(",")] 
    
   #si usa la funzione per elimianre 
    df = elimina_colonne(df, colonne_da_eliminare) 
    
   #PASSO 2: ordina la colonna ID
    df = ordina_colonna(df, "Sample code number")

    
   # PASSO 3: eliminare le righe duplicate in base a una colonna scelta dall'utente
    #colonna_scelta = input("Inserisci il nome della colonna su cui eliminare i duplicati e i valori nulli: ")
    df = elimina_duplicati_su_colonna(df, "Sample code number")
    ###################
    # PASSO 3: eliminare le righe completamente duplicate
   # df = df.drop_duplicates()  # Confronta tutte le colonne e rimuove i duplicati
    #print("Righe duplicate eliminate considerando tutte le colonne.")
    ##################
    
    # PASSO 4: convertire tutte le colonne in numeri interi
    df = converti_a_numerico(df)  
    
    # PASSO 5: correggere i numeri errati
    df = correggi_valori(df)
    
    # PASSO 6: correggere le colonne scelte dall'utente (1 diventa2)
    #colonne_class_type = input("Inserisci i nomi delle colonne in cui convertire gli 1 in 2, separati da una virgola: ")
    df = correggi_class_type(df, ["classtype_v1"])

    
 
  # PASSO 7: scambiare due colonne scelte dall'utente
   # colonna1 = input("Inserisci il nome della prima colonna da scambiare: ").strip()
    #colonna2 = input("Inserisci il nome della seconda colonna da scambiare: ").strip()
    df = scambia_colonne(df)
    
   # df salvato
    salva_file(df, output_path)
    """