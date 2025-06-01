import numpy as np
from scipy.stats import wilcoxon

# Definindo os valores de R² Original e R² Transform
r2_original = np.array([
    0.33, 0.55, 0.68, 0.41, 0.67, 0.75,
    0.40, 0.66, 0.76, 0.45, 0.71, 0.81,
    0.39, 0.63, 0.73, 0.44, 0.65, 0.76,
    0.50, 0.68, 0.78, 0.51, 0.71, 0.82
])

r2_transform = np.array([
    0.2919, 0.5404, 0.6615, 0.2542, 0.5618, 0.7051,
    0.2647, 0.6061, 0.7291, 0.3152, 0.6257, 0.7426,
    0.2529, 0.5668, 0.6880, 0.2445, 0.6084, 0.7146,
    0.3402, 0.5958, 0.7208, 0.3655, 0.6571, 0.7524
])

# Verificando o tamanho dos arrays
assert r2_original.shape == r2_transform.shape, "Os vetores r2_original e r2_transform devem ter o mesmo tamanho."

# Aplicando o teste de Wilcoxon signed-rank
stat, p_value = wilcoxon(r2_original, r2_transform, zero_method='wilcox', alternative='two-sided')

print(f"Estatística W = {stat:.3f}")
print(f"p-value = {p_value:.8f}\n")

# Contagem de sinais das diferenças
differences = r2_original - r2_transform
num_positive = np.sum(differences > 0)
num_negative = np.sum(differences < 0)
num_zero = np.sum(differences == 0)

print("Contagem de diferenças:")
print(f"  R² Original > R² Transform: {num_positive} vezes")
print(f"  R² Original < R² Transform: {num_negative} vezes")
print(f"  R² Original = R² Transform:  {num_zero} vezes")

# Interpretação básica
alpha = 0.05
if p_value < alpha:
    print("\nResultado: Rejeitamos H₀ — há diferença estatisticamente significativa entre R² Original e R² Transform.")
else:
    print("\nResultado: Falha em rejeitar H₀ — não há evidência estatística de diferença nos R².")
