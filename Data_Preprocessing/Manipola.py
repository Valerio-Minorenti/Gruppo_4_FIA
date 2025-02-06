from Data_Preprocessing.Algoritmo_manipolazione_dati import ManipolaDati

def elabora_dati(input_path, output_path):
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
    return df  # Restituisce il dataframe elaborato