import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
from pathlib import Path
from statsmodels.graphics.gofplots import qqplot
from scipy.stats.mstats import winsorize
import json
from pathlib import Path
from scipy.stats import boxcox
from scipy.signal import periodogram
from scipy.special import digamma

script_dir = Path(__file__).parent

ativos = ['ABCB4', 'FLRY3', 'GGBR3', 'ITUB3', 'PETR3', 'PRIO3', 'RADL3', 'VALE3']
for ativo in ativos:
    path_data = (
        f"data/splitted_data/train/"
        f"{ativo}/"
        f"train_price_history_{ativo}_SA_meta_dataset_ffill.csv"
    )

    # Carregar dados
    df = pd.read_csv(path_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # ============================================================
    # 1. Funções de Análise Aprimoradas
    # ============================================================
    def analisar_distribuicao_completa(series, nome):
        """Analisa distribuição com testes de normalidade e outliers"""
        print(f"\n{'='*40}\nAnálise: {nome}\n{'='*40}")
        
        # Estatísticas básicas
        skew = stats.skew(series)
        kurt = stats.kurtosis(series)
        print(f"\n--- Estatísticas Descritivas ---")
        print(f"Assimetria: {skew:.3f} | Curtose: {kurt:.3f}")
        print(f"Média: {series.mean():.3f} | Mediana: {series.median():.3f}")
        print(f"Mínimo: {series.min():.3f} | Máximo: {series.max():.3f}")
        
        # Testes de normalidade
        print("\n--- Testes de Normalidade ---")
        
        # Shapiro-Wilk
        shapiro_stat, shapiro_p = stats.shapiro(series)
        print(f"Shapiro-Wilk: p-value = {shapiro_p:.4f} {'(Normal)' if shapiro_p > 0.05 else ''}")
        
        # Anderson-Darling
        anderson_result = stats.anderson(series, dist='norm')
        print(f"Anderson-Darling: Estatística = {anderson_result.statistic:.3f}")
        for i in range(len(anderson_result.critical_values)):
            sl, cv = anderson_result.significance_level[i], anderson_result.critical_values[i]
            print(f"  Nível {sl}%: {cv:.3f} {'< Rejeita Normalidade' if anderson_result.statistic > cv else ''}")
        
        # Kolmogorov-Smirnov
        ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
        print(f"Kolmogorov-Smirnov: p-value = {ks_p:.4f} {'(Normal)' if ks_p > 0.05 else ''}")
        
        # Análise de outliers
        print("\n--- Análise de Outliers ---")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lim_inf = q1 - 1.5*iqr
        lim_sup = q3 + 1.5*iqr
        outliers = series[(series < lim_inf) | (series > lim_sup)]
        print(f"Total de outliers (IQR): {len(outliers)} ({len(outliers)/len(series):.2%})")
        
        return {
            'skew': skew,
            'kurtosis': kurt,
            'outliers': len(outliers),
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'anderson_stat': anderson_result.statistic
        }

    def plot_comparacao_distribuicoes(original, transformado, titulo, output_dir, ativo, lambda_val, origin):
        """Plota comparação de distribuições antes/depois"""
        plt.figure(figsize=(15, 6))
        
        # Histogramas
        plt.subplot(1, 2, 1)
        sns.histplot(original, kde=True, color='blue', label='Original', alpha=0.5)
        sns.histplot(transformado, kde=True, color='red', label='Transformado', alpha=0.5)
        plt.title(f'Distribuição - {titulo}')
        plt.legend()
        
        # QQ-Plots
        plt.subplot(1, 2, 2)
        qqplot(transformado, line='s', ax=plt.gca(), label='Transformado')
        plt.title('Q-Q Plot Transformado')
        
        # Salvar e fechar
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{origin}_distribuicao_{ativo}_lambda_{lambda_val:.4f}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_boxplot_comparativo(original, transformado, output_dir, ativo, origin):
        """Plota boxplots comparativos"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(y=original, color='blue')
        plt.title('Boxplot - Original')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=transformado, color='red')
        plt.title('Boxplot - Transformado')
        
        # Salvar e fechar
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{origin}_boxplot_{ativo}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    # ============================================================
    # 2. Funções de Transformação (mantidas do código anterior)
    # ============================================================
    def boxcox_transform(series_original, lambda_=None):
        """Aplica transformação Box-Cox COM normalização NBC"""
        assert (series_original > 0).all(), "Dados devem ser positivos para Box-Cox"
        
        # 2. Aplicar Box-Cox
        if lambda_ is None:
            transformed, lambda_ = stats.boxcox(series_original)
        else:
            transformed = stats.boxcox(series_original, lmbda=lambda_)
       
        return transformed, lambda_

    def optimize_lambda_boxcox(series):
        """Força redução de assimetria, ignorando escala temporariamente"""
        def objective(lambda_):
            try:
                transformed = stats.boxcox(series, lambda_)
                return abs(stats.skew(transformed))  # Foco absoluto na assimetria
            except Exception:
                return np.inf
        
        # Ampliar intervalo para permitir transformações não lineares
        result = minimize_scalar(objective, bounds=(-2, 2), method='bounded')
        return result.x

    def optimize_lambda_boxcox_mle(series):
        """
        Estima o lambda que maximiza a verossimilhança da transformação Box–Cox,
        incluindo o termo jacobiano.
        """
        n = len(series)
        sum_log = np.sum(np.log(series))

        def neg_log_likelihood(lambda_):
            try:
                y_lambda = stats.boxcox(series, lmbda=lambda_)
                ssr = np.sum((y_lambda - np.mean(y_lambda))**2)
                log_lik = - (n / 2) * np.log(ssr / n) + (lambda_ - 1) * sum_log
                return -log_lik
            except Exception:
                return np.inf

        result = minimize_scalar(neg_log_likelihood, bounds=(-2, 2), method='bounded')
        return result.x


    def dynamic_winsorize(series, window=21, lower_quantile=0.05, upper_quantile=0.95):
        """Winsoriza com base em quantis dinâmicos usando volatilidade histórica"""
        # Calcular volatilidade móvel com tratamento para divisão por zero
        rolling_vol = series.rolling(window=window).std()
        rolling_vol = rolling_vol.replace(0, np.nan).ffill()  # Remove zeros
        scaled_series = series / (rolling_vol + 1e-8)  # Evita divisão por zero
        
        # Calcular quantis dinâmicos
        q_low = scaled_series.quantile(lower_quantile)
        q_up = scaled_series.quantile(upper_quantile)
        
        # Calcular limites dinâmicos
        lower_bound = q_low * rolling_vol
        upper_bound = q_up * rolling_vol
        
        # Aplicar winsorização
        series_winsorized = series.clip(lower=lower_bound, upper=upper_bound)
        
        return series_winsorized, q_low, q_up

    def geometric_mean(series):
        return np.exp(np.mean(np.log(series)))

    def seasonal_difference(series, seasonal_period):
        return series[seasonal_period:] - series[:-seasonal_period]

    def estimate_pev(series, m=1):
        # Calcula o periodograma
        freq, I = periodogram(series, detrend='linear')
        
        # Remove componente de frequência zero (média)
        freq = freq[1:]
        I = I[1:]
        
        M = len(I)
        
        if m == 1:
            # Estimador Davis-Jones
            sigma2 = np.exp(np.mean(np.log(2 * np.pi * I)))
        else:
            # Estimador Hannan-Nicholls
            correction = digamma(m) - np.log(m)
            sigma2 = m * np.exp(np.mean(np.log(2 * np.pi * I)) - correction)
        
        return sigma2

    def optimize_lambda_gridsearch(series, seasonal_period=7, lambda_range=(-2, 2), n_points=50):
        # Passo 1: Grid de valores de lambda
        lambdas = np.linspace(lambda_range[0], lambda_range[1], n_points)
        
        # Pré-calcula a média geométrica global
        g_y = geometric_mean(series)
        
        best_lambda = None
        best_pev = np.inf
        
        for lambda_ in lambdas:
            try:
                # Passo 2: Aplicar Box-Cox
                y_lambda = boxcox(series, lmbda=lambda_)
                
                # Passo 3: Normalização com Jacobiano
                z_lambda = (g_y ** (1 - lambda_)) * y_lambda
                
                # Passo 4: Diferenciação sazonal
                u_t = seasonal_difference(z_lambda, seasonal_period)
                
                # Passo 5: Estimar PEV (usando m=1 para Davis-Jones)
                if len(u_t) >= 2 * seasonal_period:
                    current_pev = estimate_pev(u_t, m=1)
                    
                    if current_pev < best_pev:
                        best_pev = current_pev
                        best_lambda = lambda_
                        
            except Exception as e:
                continue
        
        return best_lambda

    # ============================================================
    # 3. Execução do Processo
    # ============================================================
    # Analisar dados originais
    close_series_original = df['Close'].copy()
    metrics_original  = analisar_distribuicao_completa(close_series_original, "Dados Originais")

    close_winsorized, q_low, q_up = dynamic_winsorize(close_series_original, window=21)

    # Aplicar transformação
    optimal_lambda = optimize_lambda_gridsearch(close_winsorized)
    print(f"\nLambda ótimo (com penalização de escala): {optimal_lambda:.4f}")
    df['BC_Transformed'], bc_lambda = boxcox_transform(
        close_series_original, optimal_lambda
    )

    metricas_boxcox = {
        "lambda": bc_lambda,
    }

    # Analisar dados transformados
    metrics_transformado  = analisar_distribuicao_completa(df['BC_Transformed'], "Dados Transformados com NBC")

    temp_transformed = stats.boxcox(close_series_original, lmbda=optimal_lambda)

    print("\n=== Validação Intermediária ===")
    print(f"Média após Box-Cox + NBC: {temp_transformed.mean():.2f}")
    print(f"Desvio padrão após Box-Cox + NBC: {temp_transformed.std():.2f}")

    print("\nMédia geométrica original:", np.exp(np.mean(np.log(close_series_original))))
    scale_factor = np.exp(np.mean(np.log(close_series_original))) ** (1 - optimal_lambda)
    print(f"Fator de escala: {scale_factor:.4f}")

    print("\n=== Comparação de Escala ===")
    print(f"Média Original: {close_series_original.mean():.2f}")
    print(f"Média Transformada: {df['BC_Transformed'].mean():.2f}")
    print(f"Razão Médias: {df['BC_Transformed'].mean()/close_series_original.mean():.2f}x")

    metrics_comparativo = {
        "media_BC": temp_transformed.mean(),
        "desvio_BC": temp_transformed.std(),
        "media_orig_geometrica": np.exp(np.mean(np.log(close_series_original))),
        "fator_escala": scale_factor,
        "media_original": close_series_original.mean(),
        "desvio_padrao_original": close_series_original.std(),
        "media_transformada": df['BC_Transformed'].mean(),
        "razao_medias": df['BC_Transformed'].mean() / close_series_original.mean()
    }

    # Plotar comparações
    plot_comparacao_distribuicoes(
        close_series_original, 
        df['BC_Transformed'], 
        f"Lambda: {optimal_lambda:.4f}", 
        f'src/analysis/results/{ativo}',
        ativo,
        optimal_lambda,
        'Train'
    )
    plot_boxplot_comparativo(
        close_series_original,
        df['BC_Transformed'],
        f'src/analysis/results/{ativo}',
        ativo,
        'Train'
    )

    # ============================================================
    # 4. Cálculo das Features Móveis e Salvamento (mantido)
    # ============================================================
    def calcular_estatisticas_movels(df, serie_base, janelas):
        for window in janelas:
            media_movel = df[serie_base].rolling(window=window, min_periods=1).mean().shift(1)
            desvio_movel = df[serie_base].rolling(window=window, min_periods=1).std().shift(1)
            z_score = (df[serie_base] - media_movel) / (desvio_movel + 1e-8)
            
            df[f'BC_Media_{window}'] = media_movel
            df[f'BC_Desvio_{window}'] = desvio_movel
            df[f'BC_ZScore_{window}'] = z_score

    calcular_estatisticas_movels(df, 'BC_Transformed', [7, 14, 21])

    novo_nome = f"boxcox_train_price_history_{ativo}_SA_meta_dataset_ffill.csv"
    novo_path = Path(path_data).parent / novo_nome
    df.to_csv(novo_path, index=True)

    metricas_train = {
        "ATIVO": ativo,
        "METRICAS_ORIGINAIS": metrics_original,
        "METRICAS_TRANSFORMADAS": metrics_transformado,
        "METRICAS_BOXCOX": metricas_boxcox,
        "METRICAS_COMPARATIVAS": metrics_comparativo
    }

    def salvar_metricas(metrics, ativo, origin):
        """Salva métricas em arquivo CSV"""
        output_dir = f'src/analysis/results/{ativo}'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Converter tipos numpy para python
        for k, v in metrics.items():
            if isinstance(v, (np.float32, np.float64)):
                metrics[k] = float(v)
                
        # Criar DataFrame e salvar
        df_metrics = pd.DataFrame([metrics])
        
        file_path = output_dir / f'{origin}_results_v3_{ativo}.csv'
        df_metrics.to_csv(file_path, index=False)

    salvar_metricas(metricas_train, ativo, 'Train')

    print("=== Aplicando transformação para os dados de Teste usando o mesmo lambda estimado no Treino ===")
    print("Parâmetros extraídos do TREINO:")
    print(f"  - lower_quantile (q_low): {q_low:.6f}")
    print(f"  - upper_quantile (q_up): {q_up:.6f}")
    print(f"  - lambda Yeo-Johnson (custom_lambda): {optimal_lambda:.6f}")

    path_test_data = (
        f"data/splitted_data/test/"
        f"{ativo}/"
        f"test_price_history_{ativo}_SA_meta_dataset_ffill.csv"
    )

    df_test = pd.read_csv(path_test_data)
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    df_test.set_index('Date', inplace=True)

    close_series_original_test = df_test['Close'].copy()
    metrics_original_test  = analisar_distribuicao_completa(close_series_original_test, "Dados Originais Teste")

    def dynamic_winsorize_test(series, window, q_low, q_up):
        rolling_vol = series.rolling(window).std()
        lower_bound = q_low * rolling_vol
        upper_bound = q_up  * rolling_vol
        return series.clip(lower=lower_bound, upper=upper_bound)

    winsorized_test_data = dynamic_winsorize_test(
        close_series_original_test,
        window=21,
        q_low=q_low,
        q_up=q_up
    )

    df_test['BC_Transformed'], bc_lambda = boxcox_transform(
        close_series_original_test, optimal_lambda
    )

    metricas_boxcox_test = {
        "lambda": bc_lambda,
    }

    metrics_transformado_test  = analisar_distribuicao_completa(df_test['BC_Transformed'], "Dados Teste Transformados com NBC")

    temp_transformed_test = stats.boxcox(close_series_original_test, lmbda=optimal_lambda)

    scale_factor = np.exp(np.mean(np.log(close_series_original_test))) ** (1 - optimal_lambda)

    metrics_comparativo_test = {
        "media_BC": temp_transformed_test.mean(),
        "desvio_BC": temp_transformed_test.std(),
        "media_orig_geometrica": np.exp(np.mean(np.log(close_series_original_test))),
        "fator_escala": scale_factor,
        "media_original": close_series_original_test.mean(),
        "desvio_padrao_original": close_series_original_test.std(),
        "media_transformada": df_test['BC_Transformed'].mean(),
        "razao_medias": df_test['BC_Transformed'].mean() / close_series_original_test.mean()
    }

    # Plotar comparações
    plot_comparacao_distribuicoes(
        close_series_original_test, 
        df_test['BC_Transformed'], 
        f"Lambda: {optimal_lambda:.4f}", 
        f'src/analysis/results/{ativo}',
        ativo,
        optimal_lambda,
        'Test'
    )
    plot_boxplot_comparativo(
        close_series_original_test,
        df_test['BC_Transformed'],
        f'src/analysis/results/{ativo}',
        ativo,
        'Test'
    )

    metricas_test = {
        "ATIVO": ativo,
        "METRICAS_ORIGINAIS": metrics_original_test,
        "METRICAS_TRANSFORMADAS": metrics_transformado_test,
        "METRICAS_BOXCOX": metricas_boxcox_test,
        "METRICAS_COMPARATIVAS": metrics_comparativo_test
    }
    salvar_metricas(metricas_test, ativo, 'Test')

    calcular_estatisticas_movels(df_test, 'BC_Transformed', [7, 14, 21])
    test_dir = Path(path_test_data).parent
    novo_nome_test = f"boxcox_transformed_test_price_history_{ativo}_SA_meta_dataset_ffill.csv"
    novo_path_test = test_dir / novo_nome_test

    if novo_path_test.exists():
        print(f"Arquivo de TESTE '{novo_path_test.name}' já existe. Excluindo versão antiga...")
        novo_path_test.unlink()

    df_test.to_csv(novo_path_test, index=True)
    print(f"Arquivo transformado de TESTE salvo em: {novo_path_test}")

    