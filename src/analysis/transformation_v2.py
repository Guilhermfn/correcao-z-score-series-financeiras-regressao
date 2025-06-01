import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import yeojohnson, normaltest, shapiro
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
from scipy.optimize import minimize_scalar, minimize
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from arch import arch_model

script_dir = Path(__file__).parent

ativo = 'ITUB3'
path_data = (
    f"data/splitted_data/train/"
    f"{ativo}/"
    f"train_price_history_{ativo}_SA_meta_dataset_ffill.csv"
)


df = pd.read_csv(path_data)
print(df.head(10))
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

precos = df['Close']
df['Log_Ret'] = np.nan

for i in range(1, len(df)):
    close_anterior = np.log(df['Close'].iat[i-1])
    close_atual = np.log(df['Close'].iat[i])
    df.iat[i, df.columns.get_loc('Log_Ret')] = (close_atual - close_anterior)

df.dropna(subset=['Log_Ret'], inplace=True)

# Estatísticas Descritivas
print("Estatísticas Descritivas dos Log Retornos:")
print(df['Log_Ret'].describe())

# # 2. Winsorização dinâmica aprimorada
# def dynamic_winsorize(series, window=21, lower_quantile=0.01, upper_quantile=0.99):
#     """Winsoriza com base em quantis dinâmicos usando volatilidade histórica"""
#     rolling_vol = series.rolling(window).std()
#     scaled_series = series / rolling_vol

#     q_low = scaled_series.quantile(lower_quantile)
#     q_up  = scaled_series.quantile(upper_quantile)

#     lower_bound = q_low * rolling_vol
#     upper_bound = q_up * rolling_vol

#     series_winsorized = series.clip(lower=lower_bound, upper=upper_bound)
#     return series_winsorized, q_low, q_up

# df['Log_Ret_clean'], q_low, q_up = dynamic_winsorize(df['Log_Ret'])

# print("=================================")
# print(f"Antes: {len(df['Log_Ret_clean'])} // Depois: {len(df['Log_Ret_clean'].dropna())}")
# print("=================================")

# # 3. Transformação Yeo-Johnson
# def yeojohnson_transform(series, lambda_=None):
#     if lambda_ is None:  # Usar lambda padrão do SciPy
#         transformed, lambda_ = stats.yeojohnson(series)
#     else:  # Lambda fixo
#         transformed = stats.yeojohnson(series, lambda_)
#     return transformed, lambda_

# def optimize_lambda(series, alpha=0.7):
#     """Minimiza combinação de curtose e assimetria absolutas"""
#     def objective(lambda_):
#         transformed = yeojohnson(series, lambda_)
#         skew_abs = abs(stats.skew(transformed))
#         kurt_abs = abs(stats.kurtosis(transformed))
#         return alpha*kurt_abs + (1-alpha)*skew_abs
    
#     result = minimize_scalar(objective, bounds=(-0.5, 1.5), method='bounded')
#     return result.x

# # Encontrar lambda ótimo
# custom_lambda = optimize_lambda(df['Log_Ret_clean'].dropna(), alpha=0.85)
# print(f"Lambda customizado (minimizando curtose): {custom_lambda:.4f}")

# df['YJ_Transformed'], yj_lambda = yeojohnson_transform(df['Log_Ret_clean'].dropna(), custom_lambda)

# # 4. Função de teste de normalidade
# def enhanced_normality_test(data, name):
#     """Executa bateria completa de testes"""
#     print(f"\n=== Teste de Normalidade: {name} ===")
    
#     # Testes estatísticos
#     shapiro_p = stats.shapiro(data)[1]
#     ad_stat = stats.anderson(data).statistic
    
#     # Momentos
#     skew = stats.skew(data)
#     kurt = stats.kurtosis(data)
    
#     print(f"Shapiro-Wilk: p = {shapiro_p:.3e}")
#     print(f"Anderson-Darling: statistic = {ad_stat:.2f}")
#     print(f"Assimetria: {skew:.2f} (Alvo = 0)")
#     print(f"Curtose: {kurt:.2f} (Alvo = 0)")

# # 5. Testes antes/depois
# enhanced_normality_test(df['Log_Ret_clean'].dropna(), 'Dados Winsorizados')
# enhanced_normality_test(df['YJ_Transformed'].dropna(), 'Dados Transformados')

# # 6. Visualização comparativa
# def plot_distribution_comparison(original, transformed):
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(1, 2, 1)
#     sns.histplot(original, kde=True, color='blue')
#     plt.title('Distribuição Original (Winsorizada)')
    
#     plt.subplot(1, 2, 2)
#     sns.histplot(transformed, kde=True, color='green')
#     plt.title('Distribuição após Yeo-Johnson')
    
#     plt.tight_layout()
#     plt.show()

# plot_distribution_comparison(df['Log_Ret_clean'].dropna(), df['YJ_Transformed'].dropna())

# # 7. Gráfico Q-Q comparativo
# def plot_qq_comparison(original, transformed):
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(1, 2, 1)
#     stats.probplot(original, dist="norm", plot=plt)
#     plt.title('Q-Q Plot Original')
    
#     plt.subplot(1, 2, 2)
#     stats.probplot(transformed, dist="norm", plot=plt)
#     plt.title('Q-Q Plot Transformado')
    
#     plt.tight_layout()
#     plt.show()

# plot_qq_comparison(df['Log_Ret_clean'].dropna(), df['YJ_Transformed'].dropna())

# Calculo das estatisticas moveis
print("\n\n Estatisticas Moveis: \n")

def calcular_estatisticas_movels(df, serie_base, janelas, max_lag=21):
    for window in janelas:
        # Calcula estatísticas móveis
        media_movel = df[serie_base].rolling(window=window, min_periods=1).mean().shift(1)
        desvio_movel = df[serie_base].rolling(window=window, min_periods=1).std().shift(1)
        z_score = (df[serie_base] - media_movel) / (desvio_movel + 1e-8)  # Previne divisão por zero
        
        # Adiciona ao DataFrame com nomenclatura padrão
        df[f'YJ_Media_{window}'] = media_movel
        df[f'YJ_Desvio_{window}'] = desvio_movel
        df[f'YJ_ZScore_{window}'] = z_score

        # for lag in range(1, 22):
        #     df[f'YJ_ZScore_{window}_LAG{lag}'] = df[f'YJ_ZScore_{window}'].shift(lag)


# Uso da função (assumindo que a coluna transformada se chama 'YJ_Final_v2')
calcular_estatisticas_movels(
    df=df,
    serie_base='Log_Ret',  # Substituir pelo nome real da sua coluna transformada
    janelas=[7, 14, 21]
)

# Verificação das novas colunas
print(df[[
    'Log_Ret', 
    'YJ_Media_7', 'YJ_Desvio_7', 'YJ_ZScore_7',
    'YJ_Media_14', 'YJ_Desvio_14', 'YJ_ZScore_14',
    'YJ_Media_21', 'YJ_Desvio_21', 'YJ_ZScore_21'
]].tail(10))


orig_path = Path(path_data)
base_dir = orig_path.parent

novo_nome = f"v4_transformed_train_price_history_{ativo}_SA_meta_dataset_ffill.csv"
novo_path = base_dir / novo_nome

# Verifica se o arquivo já existe; se existir, apaga para garantir substituição
if novo_path.exists():
    print(f"Arquivo {novo_path.name} já existe. Excluindo versão antiga...")
    novo_path.unlink()

# Salva o DataFrame completo (incluindo todas as colunas novas) no CSV
df.to_csv(novo_path, index=True)
print(f"Arquivo transformado salvo em: {novo_path}")

print("=== Aplicando transformação para os dados de Teste usando o mesmo lambda estimado no Treino ===")

print("Parâmetros extraídos do TREINO:")
print(f"  - lower_quantile (q_low): {q_low:.6f}")
print(f"  - upper_quantile (q_up): {q_up:.6f}")
print(f"  - lambda Yeo-Johnson (custom_lambda): {custom_lambda:.6f}")

path_test_data = (
    f"data/splitted_data/test/"
    f"{ativo}/"
    f"test_price_history_{ativo}_SA_meta_dataset_ffill.csv"
)

df_test = pd.read_csv(path_test_data)
df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test.set_index('Date', inplace=True)

# Recalcular Log_Ret para o DF de TESTE
df_test['Log_Ret'] = np.nan
for i in range(1, len(df_test)):
    prev_log = np.log(df_test['Close'].iat[i-1])
    curr_log = np.log(df_test['Close'].iat[i])
    df_test.iat[i, df_test.columns.get_loc('Log_Ret')] = (curr_log - prev_log)
df_test.dropna(subset=['Log_Ret'], inplace=True)

def dynamic_winsorize_test(series, window, q_low, q_up):
    rolling_vol = series.rolling(window).std()
    lower_bound = q_low * rolling_vol
    upper_bound = q_up  * rolling_vol
    return series.clip(lower=lower_bound, upper=upper_bound)

df_test['Log_Ret_clean'] = dynamic_winsorize_test(
    df_test['Log_Ret'],
    window=21,
    q_low=q_low,
    q_up=q_up
)

df_test['YJ_Transformed'] = yeojohnson(
    df_test['Log_Ret_clean'].dropna().values,
    custom_lambda
)

calcular_estatisticas_movels(df_test, 'Log_Ret', [7, 14, 21])


test_dir = Path(path_test_data).parent
novo_nome_test = f"v2_transformed_test_price_history_{ativo}_SA_meta_dataset_ffill.csv"
novo_path_test = test_dir / novo_nome_test

if novo_path_test.exists():
    print(f"Arquivo de TESTE '{novo_path_test.name}' já existe. Excluindo versão antiga...")
    novo_path_test.unlink()

df_test.to_csv(novo_path_test, index=True)
print(f"Arquivo transformado de TESTE salvo em: {novo_path_test}")