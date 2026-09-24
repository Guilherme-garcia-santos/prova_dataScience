# Questão 6 — Forma da distribuição e normalidade
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

from scipy.stats import skew, shapiro

variaveis = ["magnesium", "malic_acid", "proanthocyanins", "hue"]

# a) Assimetria (três cultivares em conjunto)
resultados_skew = []
for var in variaveis:
    dados = df[var].dropna()
    valor_skew = skew(dados, bias=False)
    if abs(valor_skew) < 0.5:
        classificacao = "Aproximadamente simétrica"
    elif valor_skew >= 0.5:
        classificacao = "Assimétrica à direita"
    else:
        classificacao = "Assimétrica à esquerda"
    resultados_skew.append({"Variável": var, "Skewness": valor_skew, "Classificação": classificacao})

tabela_skew = pd.DataFrame(resultados_skew)
print(tabela_skew.round(4).to_string(index=False))

# b) Histograma com curva de densidade (KDE), com média e mediana
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, var in zip(axes, variaveis):
    sns.histplot(data=df, x=var, kde=True, bins=20, color="steelblue", edgecolor="black", ax=ax)
    ax.axvline(df[var].mean(), color="red", linestyle="--", label="Média")
    ax.axvline(df[var].median(), color="green", linestyle="-", label="Mediana")
    ax.set_title(f"{var} — Histograma + KDE")
    ax.set_xlabel(var); ax.set_ylabel("Frequência"); ax.legend()
plt.tight_layout(); plt.savefig("figuras/q6_histogramas_kde.png", dpi=110); plt.show()
sns.set_theme(style="white")

# c) Média x mediana, outliers e Shapiro-Wilk
linhas = []
for var in variaveis:
    dados = df[var]
    q1, q3 = dados.quantile(0.25), dados.quantile(0.75)
    iqr = q3 - q1
    n_out = ((dados < q1 - 1.5 * iqr) | (dados > q3 + 1.5 * iqr)).sum()
    W, p = shapiro(dados)
    linhas.append({"Variável": var, "Média": dados.mean(), "Mediana": dados.median(),
                   "Outliers": n_out, "W": W, "p-valor": p,
                   "Normal (α=0,05)?": "Sim" if p >= 0.05 else "Não"})
print()
print(pd.DataFrame(linhas).round(4).to_string(index=False))
