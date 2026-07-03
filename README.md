# Predição de Uso de CPU em Máquinas Virtuais com Machine Learning

## Descrição

Este projeto implementa um modelo de **Machine Learning** para prever o consumo de CPU de uma máquina virtual utilizando dados históricos do conjunto **Bitbrains**.

O objetivo é prever o valor de **CPU usage (%) no próximo instante de tempo (t+1)** utilizando informações passadas de CPU, memória, disco e rede.

São comparados dois algoritmos de regressão:

- Regressão Linear
- Random Forest Regressor

Ao final são exibidas métricas de desempenho, importância das variáveis e gráficos comparativos entre os valores reais e previstos.

---

# Estrutura do Projeto

```
projeto/
│
├── data/
│   └── processed/
│       └── bitbrains.parquet
│
├── main.py
└── README.md
```

---

# Bibliotecas Utilizadas

- pandas
- matplotlib
- scikit-learn
- pathlib

Instalação:

```bash
pip install pandas matplotlib scikit-learn pyarrow
```

O pacote **pyarrow** é necessário para leitura do arquivo `.parquet`.

---

# Funcionamento do Código

O programa é dividido em etapas para facilitar a organização.

## 1. Carregamento dos Dados

A função:

```python
carregar_dados()
```

realiza a leitura do arquivo:

```
data/processed/bitbrains.parquet
```

e retorna um DataFrame contendo todas as medições das máquinas virtuais.

---

## 2. Preparação dos Dados

A função:

```python
preparar_dados(df)
```

executa as seguintes operações:

- seleciona uma máquina virtual específica (`vm_id = "1"`);
- ordena os dados por tempo;
- cria a variável alvo;
- cria variáveis de atraso (lags);
- remove valores ausentes.

### Variável alvo

A variável alvo corresponde ao consumo de CPU no próximo instante.

```
CPU(t+1)
```

Ela é criada utilizando:

```python
shift(-1)
```

---

## 3. Variáveis de Entrada

São utilizadas oito características como entrada do modelo.

| Variável | Descrição |
|----------|-----------|
| cpu_lag1 | CPU no instante anterior |
| cpu_lag2 | CPU dois instantes antes |
| cpu_lag3 | CPU três instantes antes |
| mem_lag1 | Memória utilizada anteriormente |
| disk_read_lag1 | Leitura de disco anterior |
| disk_write_lag1 | Escrita de disco anterior |
| net_recv_lag1 | Dados recebidos anteriormente |
| net_sent_lag1 | Dados enviados anteriormente |

---

## 4. Divisão Treino/Teste

Os dados são divididos respeitando a ordem temporal.

- 80% para treinamento
- 20% para teste

Não é realizado embaralhamento dos dados, preservando a sequência temporal da série.

---

## 5. Treinamento

São treinados dois modelos.

### Regressão Linear

Modelo estatístico utilizado como baseline.

```python
LinearRegression()
```

---

### Random Forest

Modelo baseado em árvores de decisão.

Parâmetros utilizados:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

---

## 6. Predição

Após o treinamento, cada modelo realiza a previsão da utilização futura da CPU.

```python
modelo.predict(X_test)
```

---

## 7. Avaliação

São calculadas três métricas.

### MAE

Erro absoluto médio.

Quanto menor, melhor.

---

### RMSE

Raiz do erro quadrático médio.

Penaliza erros maiores.

Quanto menor, melhor.

---

### R²

Coeficiente de determinação.

Valores próximos de **1** indicam excelente capacidade de previsão.

---

## 8. Importância das Variáveis

Para o modelo Random Forest é exibida a importância de cada variável de entrada.

Quanto maior o valor, maior sua influência na previsão.

---

## 9. Visualizações

O programa gera dois gráficos.

### Série Temporal

Mostra:

- CPU Real
- CPU Prevista

permitindo observar o comportamento do modelo ao longo do tempo.

---

### Dispersão (Real × Previsto)

Cada ponto representa uma previsão.

Quanto mais próximos da diagonal, melhor é o desempenho do modelo.

---

# Fluxo do Programa

```
Leitura dos dados
        │
        ▼
Preparação dos dados
        │
        ▼
Criação dos Lags
        │
        ▼
Divisão Treino/Teste
        │
        ▼
Treinamento
        │
        ▼
Predição
        │
        ▼
Avaliação
        │
        ▼
Gráficos
```

---

# Exemplo de Saída

Tabela com previsões:

```
CPU Real    CPU Prevista
-------------------------
12.54          12.02
13.88          13.57
15.02          14.83
...
```

Resultados dos modelos:

```
Modelo                 MAE    RMSE    R²
------------------------------------------
Regressão Linear      2.13    3.02   0.87
Random Forest         1.48    2.15   0.93
```

Importância das variáveis:

```
Variável                 Importância

cpu_lag1                    0.62
cpu_lag2                    0.14
cpu_lag3                    0.08
mem_lag1                    0.06
disk_read_lag1              0.04
disk_write_lag1             0.03
net_recv_lag1               0.02
net_sent_lag1               0.01
```

---

# Objetivo

Demonstrar a aplicação de técnicas de aprendizado de máquina para prever a utilização futura de CPU em ambientes de computação em nuvem, utilizando séries temporais extraídas do conjunto de dados Bitbrains.

A previsão antecipada do uso de CPU pode auxiliar em mecanismos de escalonamento, alocação dinâmica de recursos e balanceamento de carga em infraestruturas virtualizadas.

---

# Autor

Desenvolvido como parte de um projeto acadêmico voltado ao estudo de predição de utilização de recursos computacionais utilizando técnicas de Machine Learning.
