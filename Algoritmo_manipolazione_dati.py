import pandas as pd

#  Percorso assoluto per arrivare al file Excel
file_path = "C:/Users/Standard/Desktop/0_DATAFRAME_EXCEL.xlsx"  #  il percorso  file
df = pd.read_excel(file_path)

#Passo 1: eliminare le colonne con campi inutili
# Elimina le colonne A, B, H, si scrivono semplicemnte gli indici delle colonne e ovviamente la riga 
df = df.drop(df.columns[[0, 1, 7]], axis=1)
#messaggio che dice che sono state eliminate le colonne
print("Colonne A, B e H eliminate,")

#Passo 2: si ordinano tutti gli id, dal più basso al più alto. 
# prendo i numeri della prima colonna  e li ordino dal più basso al più alto

df = df.sort_values(by=df.columns[0]) #sort in automatico ordina in modo crescente
#messaggio per dire che sono state ordinati gli ID
print("Colonna ordinata in modo crescente")

#PASSO 3: eliminare i duplicati, confronto tutti ivalori delle righe

# serve per eliminare le righe con duplicati dalla colonna da B a J, uso gli indici di colonna da  1 a 9

df = df.drop_duplicates(subset=df.columns[1:10])
#messaggio per dire che si è eliminato i duplicati
print("Righe duplicate dalla colonna B alla J eliminate")

#PASSO 4: convertire tutti i valori nele celle in valori numerici, perché alcune non ce l'hanno e poi uniformo in modo da non avere decimali
# Converto le colonne da A a J in numeri, però devo convertire le , in . altrimenti python mi da ValueError.
for col in df.columns[:10]:  #le colonne vanno da 1 a 10.
    # Sostituisco le virgole con i punti
    df[col] = df[col].replace({',': '.'}, regex=True)
    
    # Converto in numerico
    df[col] = pd.to_numeric(df[col], errors='raise')

#PASSO 5: Alcuni numeri sono errati, infatti ci sono 999 oppure degli 0, i numeri >10 sono sbagliati e vanno convertiti in 9 e gli 0 in 1.
# Applica le modifiche per le colonne da B a J (colonne 1 a 9)
for col in df.columns[1:10]:  # Colonne da B a J, non cpnverto la A perchè ovviamente gli ID sono maggiori di 10.
    # Sostituisci con 10 se il numero è maggiore di 10, altrimenti 0 se è vuoto o NaN
    df[col] = df[col].apply(lambda x: 10 if x > 10 else (1 if pd.isna(x) else x))


#messaggio per dire che si è convertito i valori
print("Tutti gli 0 sono 1 e i >10 sono 10")

#Passo 6:  la colonna CLass Type va corretta perchè ci possono essere solo 2 valori che sono 2 e 4. quindi tutti gli 1 vanno convertiti in 2
 
# se trovo un 1 lo sostituisco con 2,  le colonne le ho chiamate con nome invece di indice

df['classtype_v1'] = df['classtype_v1'].replace(1, 2)

#messaggio per dire che sono stati cambiati gli 1 con i 2
print(" gli 1 nella colonna H sono stati sostituiti con 2")

#PAsso 7: vanno divise visivamente e logcamnte le features e le classi, quindi si  "scambiano" le colonne.

# Scambia i valori delle colonne H e J
df['classtype_v1'], df['bareNucleix_wrong'] = df['bareNucleix_wrong'], df['classtype_v1']

# Scambia anche i nomi delle colonne
df = df.rename(columns={'classtype_v1': 'bareNucleix_wrong', 'bareNucleix_wrong': 'classtype_v1'})
#messaggio
print(" colonne H e J scambiate")
#PASSO 8: ultimo controllo di eliminazione dei duplicati, però adesso si controllano gli ID e non i valori.
# Nome della colonna da controllare per duplicati
colonna_da_copiare = 'Sample code number'

# Elimina le righe con valori ugualia
df_senza_duplicati = df.drop_duplicates(subset=colonna_da_copiare, keep='first') #drop_duplicates mi controlla se ho duplicati e li droppa, è una funzione di base di pandas
#Sono finiti i processi di manipolazione quindi si può salvare un nuovo file
"""
#Si può fare un controllo finale in cui si vede se ci sono degli ID duplicati.
# Nome della colonna 
colonna_da_copiare = 'Sample code number'

# Copia i valori della colonna in una lista
valori_colonna_lista = df[colonna_da_copiare].tolist()

# Trova i valori duplicati
conta_valori = Counter(valori_colonna_lista)
duplicati = [item for item, count in conta_valori.items() if count > 1]

# Stampa i duplicati trovati
if duplicati:
    print("Valori duplicati trovati:", duplicati)
else:
    print("Nessun valore duplicato trovato.")
"""

print("Righe duplicate eliminate e nuovo file salvato!")
# Salva il file finale dell'excel
df.to_excel("Tabella_Dati_FIA.xlsx", index=False)