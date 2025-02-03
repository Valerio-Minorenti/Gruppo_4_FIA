#import pandas as pd # type: ignore
from Algoritmo_manipolazione_dati import ManipolaDati
if __name__ == "__main__":
    #input e output path che vanno messi dall'utente
    input_path = input("Inserire il percorso assoluto di input: ").strip()
    output_path = input("Inserire il percorso assoluto di output: ").strip()
    #creazioen istanza di classe
    manipolatore = ManipolaDati(input_path,output_path)
    #dopo che sono stati scritti si salvano e printano
    input_path, output_path = manipolatore.salva_percorso()

    #dopo si carica il file nel df
    df = manipolatore.carica_file(input_path)
    
    #passo 1: eliminare le colonne non utili
    colonne_input = input("Scrivi i nomi delle colonne da eliminare separate da una virgola :")
    colonne_da_eliminare = [col.strip() for col in colonne_input.split(",")]
    df = manipolatore.elimina_colonne(df,colonne_da_eliminare)

    #Passo 2: ordinare la colonna ID
    df = manipolatore.ordina_colonna(df, "Sample code number")

    #Passo 3: eliminare i duplicati basandosi sugli ID
    df = manipolatore.elimina_duplicati_su_colonna(df, "Sample code number")
    df = df.drop_duplicates()

    #Passo 4: convertire tutti i valori delle celle in numeri
    df = manipolatore.converti_a_numerico(df)

    #Passo 5: Corregge i numeri errati (i >10 o 0 ew NaN)
    df = manipolatore.correggi_valori(df)
    #Passo 6: Correggere la colonna class type che ha degli 1 invece dei 2
    df = manipolatore.correggi_class_type(df,["classtype_v1"])
    #Passo 7: "scambiare" i valori delle colonen così si hanno seprati features e type
    df= manipolatore.scambia_colonne(df)

    #Passo ultimo
    manipolatore.salva_file(df, output_path)

