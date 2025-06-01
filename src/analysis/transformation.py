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

ativo = 'VALE3'
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
# print(df['Log_Ret'].head(10))

# Estatísticas Descritivas
print("Estatísticas Descritivas dos Log Retornos:")
print(df['Log_Ret'].describe())

# 2. Winsorização dinâmica aprimorada
def dynamic_winsorize(series, window=30, n_sigmas=5):
    """Remove outliers baseado na volatilidade histórica com ajuste assimétrico"""
    rolling_std = series.rolling(window).std()
    upper = n_sigmas * rolling_std
    lower = -n_sigmas * rolling_std * 1.5  # Trata 50% mais outliers negativos
    return series.clip(lower=lower, upper=upper)

df['Log_Ret_clean'] = dynamic_winsorize(df['Log_Ret'])

# 3. Transformação híbrida com estágio adicional de potência
def hybrid_transformation(series):
    """Transformação híbrida com:
    1. Yeo-Johnson otimizado
    2. Transformação de potência
    3. Normalização quantílica adaptativa
    4. Estabilização de variância
    """
    # Estágio 1: Yeo-Johnson
    pt = PowerTransformer(method='yeo-johnson')
    yj_transformed = pt.fit_transform(series.values.reshape(-1, 1)).flatten()
    
    # Estágio 2: Transformação de potência para reduzir curtose
    power = 0.5  # Raiz quadrada para suavizar extremos
    yj_power = np.sign(yj_transformed) * np.abs(yj_transformed) ** power
    
    # Estágio 3: Normalização quantílica dinâmica
    n_quantiles = min(1000, len(yj_power)//5)
    qt = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=n_quantiles,
        subsample=10_000_000
    )
    qt_transformed = qt.fit_transform(yj_power.reshape(-1, 1)).flatten()
    
    # Estágio 4: Padronização robusta
    iqr = np.quantile(qt_transformed, 0.75) - np.quantile(qt_transformed, 0.25)
    return qt_transformed / (iqr + 1e-8)

df['YJ_Quantile'] = hybrid_transformation(df['Log_Ret_clean'])

# 4. Análise de sensibilidade com foco em curtose
lambdas = np.linspace(-2, 2, 50)
kurtosis_scores = []

for l in lambdas:
    pt = PowerTransformer(method='yeo-johnson')
    transformed = pt.fit_transform(df['Log_Ret_clean'].values.reshape(-1, 1) * l)
    kurtosis_scores.append(stats.kurtosis(transformed))

plt.figure(figsize=(10, 6))
plt.plot(lambdas, kurtosis_scores, marker='o', color='darkred')
plt.xlabel('Lambda')
plt.ylabel('Curtose')
plt.title('Impacto do Lambda na Curtose')
plt.grid(True)
plt.show()

# 5. Modelagem GARCH com distribuição t-Student e escala adequada
def garch_modeling(series):
    """Modelagem GARCH robusta com:
    1. Escala adequada dos dados
    2. Distribuição t-Student para caudas pesadas
    3. Tratamento de valores extremos
    """
    scaled_series = series * 100  # Escala recomendada
    model = arch_model(
        scaled_series,
        vol='GARCH',
        p=1,
        q=1,
        dist='StudentsT'  # Distribuição para caudas pesadas
    )
    results = model.fit(update_freq=0, disp='off')
    return results.resid / results.conditional_volatility

# Aplicação com dados transformados e escalados
df['YJ_Quantile_scaled'] = df['YJ_Quantile'] * 0.1  # Escala adicional
df['GARCH_Residuals'] = garch_modeling(df['YJ_Quantile_scaled'])

# 6. Métricas de risco prático
def calculate_var(series, alpha=0.05):
    """Calcula Value-at-Risk com ajuste de Cornish-Fisher"""
    z_score = stats.norm.ppf(alpha)
    skew = stats.skew(series)
    kurt = stats.kurtosis(series)
    adjustment = (z_score**2 - 1)*skew/6 + (z_score**3 - 3*z_score)*kurt/24
    return np.mean(series) + (z_score + adjustment) * np.std(series)

var_95 = calculate_var(df['YJ_Quantile'])
print(f"\nValue-at-Risk (95%): {var_95:.4f}")

# 7. Função de teste de normalidade atualizada
def enhanced_normality_test(data, name):
    """Testes de normalidade com métricas adicionais"""
    print(f"\n=== Teste de Normalidade: {name} ===")
    
    # Testes estatísticos
    shapiro_p = stats.shapiro(data)[1]
    ad_stat = stats.anderson(data).statistic
    jb_stat, jb_p = stats.jarque_bera(data)
    
    # Momentos
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    
    print(f"Shapiro-Wilk: p = {shapiro_p:.3e}")
    print(f"Anderson-Darling: statistic = {ad_stat:.2f}")
    print(f"Jarque-Bera: p = {jb_p:.3e}")
    print(f"\nAssimetria: {skew:.2f} (Alvo = 0)")
    print(f"Curtose: {kurt:.2f} (Alvo = 0)")
    print(f"VaR (95%): {calculate_var(data):.4f}")

# Executar testes
enhanced_normality_test(df['YJ_Quantile'], 'Transformação Híbrida')
enhanced_normality_test(df['GARCH_Residuals'].dropna(), 'Resíduos GARCH')

# 8. Visualização comparativa aprimorada
def plot_comparison(original, transformed):
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Séries temporais
    axs[0].plot(original, label='Original')
    axs[0].plot(transformed, label='Transformado')
    axs[0].set_title('Comparação Temporal')
    axs[0].legend()
    
    # Histogramas
    sns.histplot(original, kde=True, ax=axs[1], color='blue', label='Original')
    sns.histplot(transformed, kde=True, ax=axs[1], color='green', label='Transformado')
    axs[1].set_title('Distribuição Comparada')
    
    # Q-Q Plots
    qqplot(transformed, line='s', ax=axs[2])
    axs[2].set_title('Q-Q Plot Transformado')
    
    plt.tight_layout()
    plt.show()

plot_comparison(df['Log_Ret_clean'], df['YJ_Quantile'])