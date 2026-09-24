# Questão 9 — ANOVA/Kruskal-Wallis com post-hoc
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

from scipy.stats import shapiro, levene, f_oneway, kruskal, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd

col_cultivar = "cultivar"
variavel = "malic_acid"

A = df.loc[df[col_cultivar] == "Cultivar A", variavel].dropna()
B = df.loc[df[col_cultivar] == "Cultivar B", variavel].dropna()
C = df.loc[df[col_cultivar] == "Cultivar C", variavel].dropna()

# a) Pressupostos
W_A, p_A = shapiro(A)
W_B, p_B = shapiro(B)
W_C, p_C = shapiro(C)
print("Shapiro-Wilk:")
print(f"A: W = {W_A:.4f}, p = {p_A:.3e}")
print(f"B: W = {W_B:.4f}, p = {p_B:.3e}")
print(f"C: W = {W_C:.4f}, p = {p_C:.4f}")

estat_levene, p_levene = levene(A, B, C, center="median")
print(f"\nLevene: estatística = {estat_levene:.4f}, p = {p_levene:.4f}")

normalidade = p_A >= 0.05 and p_B >= 0.05 and p_C >= 0.05
variancias_homogeneas = p_levene >= 0.05

# b) Escolha do teste
if normalidade and variancias_homogeneas:
    estatistica, p_valor = f_oneway(A, B, C)
    print("\nTeste: ANOVA")
    print(f"F = {estatistica:.4f}")
else:
    estatistica, p_valor = kruskal(A, B, C)
    print("\nTeste: Kruskal-Wallis")
    print(f"H = {estatistica:.4f}")
print(f"p = {p_valor:.3e}")

# c) Post-hoc (Tukey HSD, conforme o enunciado) + conferência não paramétrica
if p_valor < 0.05:
    print("\nHá diferença significativa entre pelo menos dois cultivares.")
    tukey = pairwise_tukeyhsd(endog=df[variavel], groups=df[col_cultivar], alpha=0.05)
    print("\nTukey HSD:")
    print(tukey)

    print("\nConferência: Mann-Whitney par a par com correção de Bonferroni")
    grupos = {"Cultivar A": A, "Cultivar B": B, "Cultivar C": C}
    pares = [("Cultivar A", "Cultivar B"), ("Cultivar A", "Cultivar C"), ("Cultivar B", "Cultivar C")]
    for g1, g2 in pares:
        _, p_par = mannwhitneyu(grupos[g1], grupos[g2])
        p_bonf = min(p_par * len(pares), 1.0)
        print(f"{g1} × {g2}: p ajustado = {p_bonf:.4g} -> {'difere' if p_bonf < 0.05 else 'não difere'}")
else:
    print("\nNão há evidência suficiente de diferença entre os cultivares.")

print("\nMédias:   " + " | ".join(f"{n} = {g.mean():.4f}" for n, g in (("A", A), ("B", B), ("C", C))))
print("Medianas: " + " | ".join(f"{n} = {g.median():.4f}" for n, g in (("A", A), ("B", B), ("C", C))))

# d) Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x=col_cultivar, y=variavel, hue=col_cultivar,
            palette=["#4C72B0", "#DD8452", "#55A868"], legend=False)
sns.stripplot(data=df, x=col_cultivar, y=variavel, color="black", alpha=0.35, jitter=True)
plt.title("Teor de malic_acid por cultivar")
plt.xlabel("Cultivar"); plt.ylabel("Malic acid")
plt.tight_layout(); plt.savefig("figuras/q9_boxplot.png", dpi=110); plt.show()
