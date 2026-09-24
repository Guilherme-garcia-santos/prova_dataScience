# AV1 Data Science — Wine Recognition Dataset
# Código completo das Questões 4 a 10

# === Configuração ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from scipy import stats
from scipy.stats import (skew, shapiro, levene, pearsonr, spearmanr, kendalltau,
                         ttest_ind, f_oneway, kruskal, mannwhitneyu)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

os.makedirs("figuras", exist_ok=True)

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["cultivar"] = pd.Series(wine.target).map({0: "Cultivar A", 1: "Cultivar B", 2: "Cultivar C"})
print(df.shape)
print(df["cultivar"].value_counts().sort_index())


# ======================================================================
# Questão 4 — Amostragem estratificada e Teorema Central do Limite
# ======================================================================
# a) Amostra estratificada: 20% de cada cultivar
amostra = df.groupby("cultivar", group_keys=False).sample(frac=0.20, random_state=42)

comp = pd.DataFrame({"pop_n": df["cultivar"].value_counts(),
                     "amostra_n": amostra["cultivar"].value_counts()})
comp["pop_%"] = (comp.pop_n / comp.pop_n.sum() * 100).round(1)
comp["amostra_%"] = (comp.amostra_n / comp.amostra_n.sum() * 100).round(1)
print(comp.sort_index())
print("n da amostra =", len(amostra))

# b) Histograma de proline: população x amostra
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.histogram_bin_edges(df["proline"], bins=15)
ax.hist(df["proline"], bins=bins, density=True, alpha=0.5, label=f"População (n={len(df)})")
ax.hist(amostra["proline"], bins=bins, density=True, alpha=0.5, label=f"Amostra estratificada (n={len(amostra)})")
ax.set_title("Proline: população vs. amostra estratificada (20%)")
ax.set_xlabel("Proline (mg/L)"); ax.set_ylabel("Densidade"); ax.legend()
plt.tight_layout(); plt.savefig("figuras/q4_histograma_proline.png", dpi=110); plt.show()

print(pd.DataFrame({"População": df["proline"].describe(),
                    "Amostra": amostra["proline"].describe()}).round(1))

# c) 1.000 reamostragens com reposição e comparação com o TCL
n = len(amostra)
rng = np.random.default_rng(42)
medias = np.array([rng.choice(df["proline"], size=n, replace=True).mean() for _ in range(1000)])

sigma = df["proline"].std(ddof=0)
ep_teorico = sigma / np.sqrt(n)
ep_obs = medias.std(ddof=1)
print(f"sigma populacional = {sigma:.2f}")
print(f"erro-padrão teórico (sigma/raiz(n)) = {ep_teorico:.2f}")
print(f"desvio-padrão observado das médias = {ep_obs:.2f}")
print(f"sigma / EP teórico = {sigma/ep_teorico:.2f} | sigma / EP observado = {sigma/ep_obs:.2f}")
print(f"média das médias = {medias.mean():.1f} | média populacional = {df['proline'].mean():.1f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(medias, bins=30, edgecolor="white")
ax.axvline(df["proline"].mean(), color="red", ls="--", label="Média populacional")
ax.set_title(f"Distribuição de 1.000 médias amostrais de proline (n={n})")
ax.set_xlabel("Média de proline (mg/L)"); ax.set_ylabel("Frequência"); ax.legend()
plt.tight_layout(); plt.savefig("figuras/q4_medias_tcl.png", dpi=110); plt.show()


# ======================================================================
# Questão 5 — Medidas de posição, dispersão e outliers
# ======================================================================
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


# ======================================================================
# Questão 6 — Forma da distribuição e normalidade
# ======================================================================
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


# ======================================================================
# Questão 7 — Correlação, significância e redundância
# ======================================================================
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


# ======================================================================
# Questão 8 — Teste de hipótese entre dois grupos
# ======================================================================
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


# ======================================================================
# Questão 9 — ANOVA/Kruskal-Wallis com post-hoc
# ======================================================================
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


# ======================================================================
# Questão 10 — Questão integradora
# ======================================================================
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
