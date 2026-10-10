# Classificação de Fake News em Português (Fake.br)

Modelo de classificação supervisionado que identifica notícias **verdadeiras (REAL)** ou **falsas (FAKE)** a partir do corpus [Fake.br](https://github.com/roneysco/Fake.br-Corpus), usando NLTK e Scikit-Learn.

> Trabalho acadêmico da disciplina de Processamento de Linguagem Natural (NLP).

## Resultado

| Item | Valor |
|---|---|
| Modelo | Regressão logística (`solver='lbfgs'`) |
| Representação | TF-IDF com n-gramas de 1 a 3 palavras |
| Divisão treino/teste | 75% / 25% (estratificada, `random_state=42`) |
| **Acurácia no teste** | **90,44%** |

O corpus balanceado tem 3.600 notícias FAKE e 3.600 REAL, com o mesmo total de palavras em cada classe (359.028).

### N-gramas usados no treino (2.700 textos por classe)

| Classe | Tipo | Distintos | No vocabulário TF-IDF |
|---|---|---|---|
| REAL | Palavras | 13.236 | 9.316 |
| REAL | Bigramas | 176.573 | 44.597 |
| REAL | Trigramas | 235.296 | 14.148 |
| FAKE | Palavras | 12.826 | 9.409 |
| FAKE | Bigramas | 185.370 | 46.879 |
| FAKE | Trigramas | 244.503 | 14.981 |

O vocabulário TF-IDF é compartilhado entre as classes e tem 92.920 termos.

## Pipeline

1. **Carga** do `pre-processed.csv` do corpus em um DataFrame pandas (UTF-8).
2. **Limpeza** dos textos:
   - tokenização (`word_tokenize`, NLTK);
   - minúsculas, remoção de acentos e números;
   - remoção de pontuação, símbolos e stopwords (NLTK, português);
   - stemming com o RSLP.
3. **Truncamento dos pares:** cada notícia REAL é cortada para o número de palavras da FAKE pareada. Sem isso, o modelo aprende pelo *tamanho* do texto (REAL tem cerca de 6x mais palavras) e a acurácia sobe de forma artificial.
4. **Divisão** em treino (75%) e teste (25%), estratificada.
5. **TF-IDF** com 1 a 3 gramas (`min_df=2`, `max_df=0.9`, `sublinear_tf=True`), ajustado só no treino.
6. **Regressão logística** treinada no conjunto de treino.
7. **Avaliação:** predição com `predict_proba` e acurácia com `accuracy_score`.
8. **Nuvens de palavras** das classes REAL e FAKE.

## Estrutura do repositório

```
.
├── trabalho_nlp_fakebr.ipynb   # notebook completo (código + saídas)
├── nuvem_real_<RU>.png         # nuvem de palavras dos textos verdadeiros
├── nuvem_fake_<RU>.png         # nuvem de palavras dos textos falsos
└── README.md
```

Ao rodar o notebook, ele baixa sozinho o `pre-processed.csv`, o `funcoes_auxiliares.py` e as máscaras das nuvens (pasta `data/img`).

## Como executar

Requisitos: Python 3.9+ e acesso à internet na primeira execução.

```bash
pip install matplotlib wordcloud nltk scikit-learn pandas numpy pillow jupyter
jupyter notebook trabalho_nlp_fakebr.ipynb
```

Rode as células em ordem (*Run All*). Também funciona no Google Colab.

## Tecnologias

Python · pandas · NumPy · NLTK · Scikit-Learn · WordCloud · Matplotlib

## Observações

- A acurácia pode variar um pouco se o pré-processamento, o `random_state` ou os parâmetros do TF-IDF forem alterados.
- Acurácias acima de 93% costumam indicar uso de textos completos, sem normalizar o tamanho: no teste de controle sem truncamento, o mesmo pipeline chega a cerca de 97%.

## Créditos

- Corpus **Fake.br**: Monteiro et al., *Contributions to the Study of Fake News in Portuguese: New Corpus and Automatic Detection Results*, PROPOR 2018. Repositório: <https://github.com/roneysco/Fake.br-Corpus>
- Funções auxiliares e máscaras das nuvens: <https://github.com/N-CPUninter/NLP>
