# Questão 4 — Amostragem estratificada e Teorema Central do Limite
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
