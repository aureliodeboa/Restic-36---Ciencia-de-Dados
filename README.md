# Projeto de Regressão Linear: Análise de Engajamento de Influenciadores no Instagram

Este projeto tem como objetivo analisar e prever o engajamento de influenciadores no Instagram utilizando um modelo de regressão linear. A base de dados utilizada contém informações sobre o número de seguidores, posts, curtidas, e taxa de engajamento, entre outras variáveis.

## Descrição do Projeto

A partir da base de dados dos influenciadores do Instagram, o projeto busca analisar a relação entre diversas variáveis e a taxa de engajamento dos influenciadores. O modelo de regressão linear foi utilizado para entender essa relação e realizar previsões sobre a taxa de engajamento com base em variáveis como o número de seguidores, curtidas por post e a média de curtidas em posts recentes.

### Objetivos

- Analisar a correlação entre variáveis como seguidores, curtidas e número de postagens.
- Construir e treinar um modelo de regressão linear para prever a taxa de engajamento.
- Comparar o modelo de regressão linear com os modelos de regularização Lasso e Ridge.
- Avaliar a precisão do modelo utilizando métricas como MSE, MAE, e R².

## Base de Dados

A base de dados utilizada foi baixada do Kaggle e contém informações sobre influenciadores no Instagram. O dataset inclui as seguintes colunas principais:

- **followers**: Número de seguidores do influenciador.
- **posts**: Número de postagens feitas pelo influenciador.
- **avg_likes**: Média de curtidas por post.
- **60_day_eng_rate**: Taxa de engajamento dos últimos 60 dias.
- **new_post_avg_like**: Média de curtidas nos novos posts.

O dataset foi limpo para remover dados nulos e valores inconsistentes.

## 🛠️ Instruções para Replicar o Projeto

1. Clone o Repositório
Primeiro, faça o clone do repositório em sua máquina local:

```bash
Copiar código
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```
2. Configure o Ambiente
Certifique-se de que possui o Python 3.7 ou superior instalado. Recomenda-se a utilização de um ambiente virtual para instalar as dependências:

```bash
Copiar código
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as Dependências
Instale todas as bibliotecas necessárias listadas no arquivo requirements.txt:

```bash
pip install -r requirements.txt
```
Alternativamente você pode instalar as dependências manualmente:

```bash
pip install kagglehub pandas seaborn numpy matplotlib scikit-learn
```

### Pré-requisitos

Certifique-se de ter os seguintes pacotes Python instalados:

- `pandas`
- `seaborn`
- `numpy`
- `matplotlib`
- `sklearn`
- `kagglehub`

### Passos

1. Baixe o dataset utilizando o KaggleHub:

    ```python
    import kagglehub
    path = kagglehub.dataset_download("surajjha101/top-instagram-influencers-data-cleaned")
    ```

2. Carregue o dataset e realize uma análise exploratória dos dados:

    ```python
    import pandas as pd
    df = pd.read_csv('path/to/dataset.csv')
    ```

3. Limpeza e transformação dos dados, como remover valores nulos e converter unidades:

    ```python
    df = df.dropna(subset=['country'])
    ```

4. Visualize a correlação entre as variáveis com um gráfico de dispersão e matriz de correlação:

    ```python
    import seaborn as sns
    sns.heatmap(df.corr(), annot=True)
    ```

5. Normalize os dados e divida o conjunto de dados entre treino e teste:

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

6. Treine o modelo de regressão linear e calcule as métricas de avaliação:

    ```python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```

7. Avalie o modelo com as métricas MSE, MAE e R².

    ```python
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ```

### Resultados

- **MSE**: [Valor]
- **MAE**: [Valor]
- **R²**: [Valor]

Os resultados mostram que o modelo de regressão linear conseguiu explicar uma parte significativa da variabilidade na taxa de engajamento dos influenciadores com uma boa precisão.

## Relatório

O relatório detalhado deste projeto, incluindo a metodologia, análise de dados e resultados obtidos, pode ser encontrado no arquivo PDF [relatorio.pdf](./relatorio.pdf).

## Conclusão

O modelo de regressão linear forneceu uma boa base para análise de tendências no engajamento de influenciadores do Instagram. Embora o modelo tenha mostrado bons resultados, outras variáveis, como o conteúdo dos posts e a frequência de postagens, podem influenciar ainda mais a taxa de engajamento e poderiam ser exploradas em modelos mais complexos.

---

**Autor**: Andressa Carvalho, Aurélio José
