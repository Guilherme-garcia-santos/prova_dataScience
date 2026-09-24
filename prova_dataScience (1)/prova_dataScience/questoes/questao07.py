# Questão 7 — Correlação, significância e redundância
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

from scipy import stats
from scipy.stats import pearsonr, shapiro, spearmanr, kendalltau

col_cultivar = "cultivar"
variaveis = wine.feature_names          # as 13 variáveis

# a) Matriz de correlação e pares com |r| > 0,7
matriz_corr = df[variaveis].corr(method="pearson")
print("ITEM (A) — MATRIZ DE CORRELAÇÃO")
print(matriz_corr.round(2))

pares = []
for i in range(len(variaveis)):
    for j in range(i + 1, len(variaveis)):
        var1, var2 = variaveis[i], variaveis[j]
        r = matriz_corr.loc[var1, var2]
        par_excluido = {var1, var2} == {"flavanoids", "total_phenols"}
        if abs(r) > 0.7 and not par_excluido:
            pares.append({"Variável 1": var1, "Variável 2": var2, "r": r, "|r|": abs(r)})

tabela_pares = pd.DataFrame(pares).sort_values(by="|r|", ascending=False)
print("\nPares com |r| > 0,7:")
print(tabela_pares.round(4).to_string(index=False))

# b) Estatística t manual x scipy.stats.pearsonr
var1 = tabela_pares.iloc[0]["Variável 1"]
var2 = tabela_pares.iloc[0]["Variável 2"]
dados_par = df[[var1, var2]].dropna()
x, y = dados_par[var1], dados_par[var2]
n = len(dados_par)

r_matriz = matriz_corr.loc[var1, var2]
t_manual = r_matriz * np.sqrt(n - 2) / np.sqrt(1 - r_matriz**2)
p_manual = 2 * stats.t.sf(abs(t_manual), df=n - 2)
r, p_pearson = pearsonr(x, y)

print(f"\nITEM (B) — Par de maior |r|: {var1} × {var2} (n = {n})")
print(f"t manual = {t_manual:.4f}")
print(f"p manual (distribuição t, {n-2} g.l.) = {p_manual:.3e}")
print(f"pearsonr: r = {r:.4f}, p = {p_pearson:.3e}")

alpha = 0.05
if p_pearson < alpha:
    print("p < 0,05: rejeita-se H0 de ausência de correlação linear.")
else:
    print("p >= 0,05: não se rejeita H0 de ausência de correlação linear.")

# c) Normalidade e correlações não paramétricas
W1, p1 = shapiro(x)
W2, p2 = shapiro(y)
print("\nITEM (C) — Shapiro-Wilk")
print(f"{var1}: W = {W1:.4f}, p = {p1:.3e} -> {'rejeita' if p1 < 0.05 else 'não rejeita'} normalidade")
print(f"{var2}: W = {W2:.4f}, p = {p2:.3e} -> {'rejeita' if p2 < 0.05 else 'não rejeita'} normalidade")

rho_spearman, p_spearman = spearmanr(x, y)
tau_kendall, p_kendall = kendalltau(x, y)
print(f"\nPearson  r   = {r:.4f}")
print(f"Spearman rho = {rho_spearman:.4f} (p = {p_spearman:.3e})")
print(f"Kendall  tau = {tau_kendall:.4f} (p = {p_kendall:.3e})")

# d) Heatmap e scatter plot
plt.figure(figsize=(13, 10))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
plt.title("Matriz de Correlação de Pearson — 13 variáveis")
plt.tight_layout(); plt.savefig("figuras/q7_heatmap.png", dpi=100); plt.show()

plt.figure(figsize=(9, 7))
sns.scatterplot(data=df, x=var1, y=var2, hue=col_cultivar, palette="Set1", s=70, alpha=0.8)
sns.regplot(data=df, x=var1, y=var2, scatter=False, color="black",
            line_kws={"linewidth": 2, "label": "Reta ajustada"})
plt.title(f"{var1} × {var2}\nPearson r = {r:.3f}")
plt.xlabel(var1); plt.ylabel(var2); plt.legend()
plt.tight_layout(); plt.savefig("figuras/q7_scatter.png", dpi=110); plt.show()
