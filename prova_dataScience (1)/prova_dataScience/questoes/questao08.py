# Questão 8 — Teste de hipótese entre dois grupos
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

os.makedirs("figuras", exist_ok=True)

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["cultivar"] = pd.Series(wine.target).map({0: "Cultivar A", 1: "Cultivar B", 2: "Cultivar C"})

from scipy.stats import shapiro, levene, ttest_ind, kruskal, mannwhitneyu

col_cultivar = "cultivar"
variavel = "magnesium"

# a) Separar os grupos
B = df.loc[df[col_cultivar] == "Cultivar B", variavel].dropna()
C = df.loc[df[col_cultivar] == "Cultivar C", variavel].dropna()

for nome, g in (("Cultivar B", B), ("Cultivar C", C)):
    print(f"{nome}: N = {len(g)} | Média = {g.mean():.3f} | Mediana = {g.median():.3f} | DP = {g.std(ddof=1):.3f}")

# b) Pressupostos
W_B, p_B = shapiro(B)
W_C, p_C = shapiro(C)
print("\nShapiro-Wilk:")
print(f"B: W = {W_B:.4f}, p = {p_B:.3e}")
print(f"C: W = {W_C:.4f}, p = {p_C:.4f}")

estat_levene, p_levene = levene(B, C, center="median")
print(f"\nLevene: estatística = {estat_levene:.4f}, p = {p_levene:.4f}")

# c) Escolha do teste
if p_B >= 0.05 and p_C >= 0.05 and p_levene >= 0.05:
    estatistica, p_valor = ttest_ind(B, C, equal_var=True)
    teste = "Teste t de Student"
else:
    estatistica, p_valor = kruskal(B, C)
    teste = "Kruskal-Wallis"

print(f"\nTeste utilizado: {teste}")
print(f"Estatística = {estatistica:.4f}, p = {p_valor:.4f}")
u, p_u = mannwhitneyu(B, C)
print(f"Conferência (Mann-Whitney): U = {u:.1f}, p = {p_u:.4f}")

if p_valor < 0.05:
    print("Conclusão: existe diferença estatisticamente significativa entre os cultivares.")
else:
    print("Conclusão: não há evidência suficiente de diferença entre os cultivares.")

# d) Boxplot
dados_boxplot = df[df[col_cultivar].isin(["Cultivar B", "Cultivar C"])][[col_cultivar, variavel]]
plt.figure(figsize=(8, 6))
sns.boxplot(data=dados_boxplot, x=col_cultivar, y=variavel, hue=col_cultivar,
            palette=["#DD8452", "#55A868"], legend=False)
sns.stripplot(data=dados_boxplot, x=col_cultivar, y=variavel, color="black", alpha=0.4, jitter=True)
plt.title("Teor de magnesium — Cultivares B e C")
plt.xlabel("Cultivar"); plt.ylabel("Magnesium")
plt.tight_layout(); plt.savefig("figuras/q8_boxplot.png", dpi=110); plt.show()
