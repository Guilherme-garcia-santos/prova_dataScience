# Questão 5 — Medidas de posição, dispersão e outliers
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

col_cultivar = "cultivar"
variaveis = wine.feature_names

# a) Média, mediana, desvio-padrão e CV por cultivar
resultados = []
for cultivar, grupo in df.groupby(col_cultivar):
    for var in variaveis:
        dados = grupo[var].dropna()
        media = dados.mean()
        mediana = dados.median()
        desvio_padrao = dados.std(ddof=1)          # desvio-padrão amostral
        cv = (desvio_padrao / media) * 100         # coeficiente de variação (%)
        resultados.append({"Cultivar": cultivar, "Variável": var, "Média": media,
                           "Mediana": mediana, "Desvio-padrão": desvio_padrao, "CV (%)": cv})

tabela = pd.DataFrame(resultados).round(2)
print(tabela.to_string(index=False))

cv_medio = tabela.groupby("Variável")["CV (%)"].mean().sort_values(ascending=False)
print("\nCV médio entre os três cultivares (%):")
print(cv_medio.round(2))
top3 = cv_medio.index[:3].tolist()
print("\nTrês variáveis com maior CV médio:", top3)

# b) Outliers pela regra de 1,5×IQR, calculados dentro de cada cultivar
def conta_outliers(serie):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((serie < lim_inf) | (serie > lim_sup)).sum()

outliers = df.groupby(col_cultivar)[top3].agg(conta_outliers)
print("\nOutliers (1,5×IQR) por cultivar:")
print(outliers)

# c) Boxplots comparativos
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, var in zip(axes, top3):
    sns.boxplot(data=df, x=col_cultivar, y=var, hue=col_cultivar, ax=ax, legend=False)
    ax.set_title(f"{var} por cultivar")
    ax.set_xlabel("Cultivar"); ax.set_ylabel(var)
plt.tight_layout(); plt.savefig("figuras/q5_boxplots.png", dpi=110); plt.show()

# d) Cultivar com menor CV (mais homogêneo) em cada variável
cv_largo = tabela.pivot(index="Cultivar", columns="Variável", values="CV (%)")
mais_homogeneo = cv_largo.idxmin()
print("\nCultivar mais homogêneo (menor CV) em cada variável:")
print(mais_homogeneo.to_string())
print("\nContagem:")
print(mais_homogeneo.value_counts().to_string())
