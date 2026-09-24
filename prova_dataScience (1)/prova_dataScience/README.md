# prova_dataScience

Avaliação AV1 de Data Science — Universidade Positivo (Prof. Leandro Escobar).
Governança de dados, estatística descritiva e inferencial aplicadas ao Wine Recognition Dataset
(FORINA et al., 1991), carregado com `sklearn.datasets.load_wine`.

## Estrutura

```
prova_dataScience/
├── README.md
├── requirements.txt
├── prova_completa.ipynb # notebook único com as Questões 4 a 10
├── prova_completa.py    # mesmo código em um único script
├── questoes/            # um script por questão prática (4 a 10)
├── figuras/             # gráficos gerados pelos scripts
├── saidas/              # saídas numéricas de cada script
└── prompts_IA.txt       # prompts de IA utilizados, por questão
```

| Arquivo | Conteúdo |
|---|---|
| `questao04.py` | Amostragem estratificada (20%) e Teorema Central do Limite |
| `questao05.py` | Média, mediana, desvio-padrão, CV, outliers (1,5×IQR) e boxplots |
| `questao06.py` | Assimetria, histogramas com KDE e Shapiro-Wilk |
| `questao07.py` | Matriz de correlação, estatística t, Spearman, Kendall, heatmap e scatter |
| `questao08.py` | Magnesium: Cultivar B × Cultivar C (Shapiro, Levene, Kruskal-Wallis) |
| `questao09.py` | Malic acid entre os três cultivares (Kruskal-Wallis + Tukey HSD) |
| `questao10.py` | Questão integradora: amostra de 60%, redundância e teor alcoólico (ANOVA) |

## Como executar

```bash
pip install -r requirements.txt
python prova_completa.py          # todas as questões
python questoes/questao04.py      # ou uma questão isolada
```

Ou abra `prova_completa.ipynb` no Jupyter e execute as células em ordem.

Execute a partir da raiz do repositório; as figuras são salvas em `figuras/`.
Os cultivares são rotulados como Cultivar A, Cultivar B e Cultivar C (target 0, 1 e 2).

## Uso de IA generativa

Conforme as regras da avaliação, os prompts utilizados estão em `prompts_IA.txt`, indicando a questão de cada um.
