# :chart_with_upwards_trend: Demand Forecasting :chart_with_upwards_trend:

## :bulb: Referências e Requerimentos

- Obtive os datasets através do Kaggle: [Store Item Demand](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) e [Food Demand](https://www.kaggle.com/datasets/arashnic/food-demand);

- Utilizei a linguagem [Python](https://docs.python.org/3/), o [Kedro](https://kedro.org/) para gerenciar o projeto e o [Poetry](https://python-poetry.org/) para gerenciar o ambiente (dependências).

## :pushpin: Introdução

A previsão de demanda é uma atividade essencial para a gestão eficaz da cadeia de suprimentos. Através da previsão da demanda futura, as empresas podem otimizar o gerenciamento de estoques, planejar a produção e distribuição de produtos, e tomar decisões estratégicas mais acertadas. Nos últimos anos, a aplicação de Inteligência Artificial (IA), e em particular, de modelos de séries temporais, tem revolucionado a forma como as organizações abordam a previsão de demanda. Sistemas de aprendizado de máquina e algoritmos avançados permitem lidar com grandes volumes de dados, proporcionando insights valiosos para aprimorar as estratégias de suprimento. Este estudo tem como objetivo avaliar o desempenho dos modelos de séries temporais na previsão de demanda na gestão da cadeia de suprimentos. Para isso, será realizada uma comparação entre diferentes modelos, incluindo modelos tradicionais, como os Modelos Exponenciais, os modelos ARIMA e a Média Móvel Simples, e modelos mais avançados, como o Prophet, o XGBoost, o N-HiTS, o FourTheta e o TiDE. Além deles, também será utilizado um modelo para demandas intermitentes, o Croston. Os modelos serão treinados utilizando o Optuna para otimização dos hiperparâmetros e avaliados utilizando técnicas de validação cruzada, como a Janela de Expansão, e métricas de desempenho adequadas para séries temporais, como a Média Ponderada Geral (OWA) baseada no Erro Percentual Absoluto Médio Simétrico (sMAPE) e no Erro Logarítmico Quadrático Médio (RMSLE). Os resultados desse estudo fornecerão insights valiosos para as organizações na escolha dos modelos de previsão de demanda mais adequados para suas necessidades específicas.

## ⚙️ Como executar

### 1. No terminal/cmd, deve-se clonar o repositório:
```
git clone https://github.com/lborgatow/demand-forecasting.git
```
ou baixar o projeto diretamente do [GitHub](https://github.com/lborgatow/demand-forecasting);

### 2. Configurar o Poetry para gerar o ".venv" na pasta do projeto:
```
poetry config virtualenvs.in-project true
```

### 3. Instalar as dependências usando o Poetry:
```
poetry install --no-root
```

### 4. Executar
#### 4.1. Projeto no Kedro
```
poetry shell
```

##### 4.1.1. Base 1 [Store Item Demand]: Diária
```
$env:DATABASE="STORE_ITEM"; $env:FREQUENCY="DAILY"; kedro run      (Windows)
export DATABASE="STORE_ITEM"; export FREQUENCY="DAILY"; kedro run  (Linux)
```

##### 4.1.2. Base 1 [Store Item Demand]: Semanal
```
$env:DATABASE="STORE_ITEM"; $env:FREQUENCY="WEEKLY"; kedro run      (Windows)
export DATABASE="STORE_ITEM"; export FREQUENCY="WEEKLY"; kedro run  (Linux)
```

##### 4.1.3. Base 2 [Food Demand]: Semanal
```
$env:DATABASE="FOOD"; $env:FREQUENCY="WEEKLY"; kedro run      (Windows)
export DATABASE="FOOD"; export FREQUENCY="WEEKLY"; kedro run  (Linux)
```

#### 4.2. Notebooks 
```
Adicionar o ".venv" como kernel do notebook desejado e, em seguida, executá-lo.
```


