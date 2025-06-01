import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
import seaborn as sns
import statsmodels . api as sm
from statsmodels . graphics . tsaplots import plot_acf , plot_pacf
from pathlib import Path
script_dir = Path(__file__).parent


path_data = 'data/splitted_data/train/VALE3/train_price_history_VALE3_SA_meta_dataset_ffill.csv'
df = pd.read_csv(path_data)
print(df.head(10))
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# datas = df.index  # Use df.index instead of df['Date']
# print(datas)

# 2. Visualizar as primeiras 10 linhas
print("Primeiras 10 linhas:")
print(df.head(10))

# 3. Listar todas as colunas
print("\nColunas do dataset:")
print(df.columns.tolist())

# 4. Número de linhas e colunas
n_linhas, n_colunas = df.shape
print(f"\nDimensão do dataset: {n_linhas} linhas e {n_colunas} colunas")

# 5. Informações gerais (tipos, não-nulos, memória)
print("\nInformações gerais do DataFrame:")
df.info()

# 6. Estatísticas descritivas das colunas numéricas
print("\nEstatísticas descritivas (numéricas):")
print(df.describe())

print(df.dtypes)

# 7. Estatísticas descritivas das colunas categóricas
print("\nEstatísticas descritivas (categóricas):")
print(df.describe(include='all'))


precos = df['Close']

plt.figure(figsize=(12, 6))
plt.plot(precos.index, precos.values, color='steelblue', linewidth=1.5)
plt.title('Preços de Fechamento Diários', fontsize=14)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Preço (R$)', fontsize=12)
plt.grid(alpha=0.3)

# Salva o arquivo (opcionalmente ajustando o bbox para não cortar nada)
plt.savefig('precos_temporais.png', dpi=300, bbox_inches='tight')

plt.show()
