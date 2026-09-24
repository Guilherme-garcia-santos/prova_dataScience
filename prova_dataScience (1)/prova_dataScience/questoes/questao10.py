# Questão 10 — Questão integradora
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

from scipy.stats import shapiro, levene, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

cultivares = ["Cultivar A", "Cultivar B", "Cultivar C"]

# a) Amostra estratificada: 60% de cada cultivar
print("Tamanho da população:")
print(df["cultivar"].value_counts().sort_index().to_string())

amostra = df.groupby("cultivar", group_keys=False).sample(frac=0.60, random_state=7)

print("\nTamanho da amostra:")
print(amostra["cultivar"].value_counts().sort_index().to_string())

variaveis = wine.feature_names
rep = pd.DataFrame({"População": df[variaveis].mean(), "Amostra": amostra[variaveis].mean()})
rep["Diferença (%)"] = (rep["Amostra"] / rep["População"] - 1) * 100
print("\nMédias: população x amostra")
print(rep.round(2))

# b) Matriz de correlação na amostra e par mais redundante
matriz_corr = amostra[variaveis].corr()

pares = []
for i in range(len(variaveis)):
    for j in range(i + 1, len(variaveis)):
        v1, v2 = variaveis[i], variaveis[j]
        r = matriz_corr.loc[v1, v2]
        pares.append({"Variável 1": v1, "Variável 2": v2, "r": r, "|r|": abs(r)})
pares = pd.DataFrame(pares).sort_values("|r|", ascending=False)

pares_excluidos = [
    {"flavanoids", "total_phenols"},                    # par analisado em aula
    {"flavanoids", "od280/od315_of_diluted_wines"},     # par da Questão 7
]
pares_validos = pares[~pares.apply(
    lambda linha: {linha["Variável 1"], linha["Variável 2"]} in pares_excluidos, axis=1)]

print("\nPares mais correlacionados (excluindo aula e Questão 7):")
print(pares_validos.head(5).round(4).to_string(index=False))

par = pares_validos.iloc[0]
print(f"\nDuas variáveis mais redundantes: {par['Variável 1']} × {par['Variável 2']} (r = {par['r']:.4f})")

plt.figure(figsize=(13, 10))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
plt.title("Matriz de correlação — amostra estratificada de 60%")
plt.tight_layout(); plt.savefig("figuras/q10_heatmap.png", dpi=100); plt.show()

# c) Teor de alcohol entre os três cultivares (na amostra)
A, B, C = [amostra.loc[amostra["cultivar"] == c, "alcohol"].dropna() for c in cultivares]

W_A, p_A = shapiro(A)
W_B, p_B = shapiro(B)
W_C, p_C = shapiro(C)
print("\nShapiro-Wilk:")
print(f"A: W = {W_A:.4f}, p = {p_A:.4f}")
print(f"B: W = {W_B:.4f}, p = {p_B:.4f}")
print(f"C: W = {W_C:.4f}, p = {p_C:.4f}")

estat_levene, p_levene = levene(A, B, C, center="median")
print(f"\nLevene: estatística = {estat_levene:.4f}, p = {p_levene:.4f}")

normalidade = p_A >= 0.05 and p_B >= 0.05 and p_C >= 0.05
variancias_homogeneas = p_levene >= 0.05

if normalidade and variancias_homogeneas:
    estatistica, p_valor = f_oneway(A, B, C)
    teste = "ANOVA"
else:
    estatistica, p_valor = kruskal(A, B, C)
    teste = "Kruskal-Wallis"

print(f"\nTeste escolhido: {teste}")
print(f"Estatística = {estatistica:.4f}, p-valor = {p_valor:.3e}")
print("Médias:   " + " | ".join(f"{n} = {g.mean():.4f}" for n, g in (("A", A), ("B", B), ("C", C))))
print("Medianas: " + " | ".join(f"{n} = {g.median():.4f}" for n, g in (("A", A), ("B", B), ("C", C))))

if p_valor < 0.05:
    print("Há evidência estatística de diferença no teor de alcohol entre pelo menos dois cultivares.")
    # Complemento (não exigido): quais pares diferem, relevante para preço por cultivar
    print(pairwise_tukeyhsd(endog=amostra["alcohol"], groups=amostra["cultivar"], alpha=0.05))
else:
    print("Não há evidência estatística suficiente de diferença no teor de alcohol.")

plt.figure(figsize=(8, 6))
sns.boxplot(data=amostra, x="cultivar", y="alcohol", hue="cultivar", order=cultivares, legend=False)
plt.title("Teor alcoólico por cultivar — amostra de 60%")
plt.xlabel("Cultivar"); plt.ylabel("Alcohol (% vol)")
plt.tight_layout(); plt.savefig("figuras/q10_boxplot.png", dpi=110); plt.show()
