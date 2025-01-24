
import pandas as pd

# Legge il file CSV, i due percorsi sono dei file
csv_file = 'C:/Users/Standard/Desktop/tabella1.csv'  
excel_file = 'C:/Users/Standard/Desktop/versione_2.xlsx'  

# Pandas per poter manipolare il file excel
df = pd.read_csv(csv_file)
df.to_excel(excel_file, index=False, engine='openpyxl')  
