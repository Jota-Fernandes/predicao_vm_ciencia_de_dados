from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================
# CARREGAMENTO DOS DADOS
# =====================================================

def carregar_dados():
    base_dir = Path(__file__).resolve().parent

    df = pd.read_parquet(
        base_dir / "data" / "processed" / "bitbrains.parquet"
    )

    return df


# =====================================================
# PREPARAÇÃO DOS DADOS
# =====================================================

def preparar_dados(df, vm_id="1"):

    vm = (
        df[df["vm_id"] == vm_id]
        .sort_values("Timestamp [ms]")
        .copy()
    )

    # ==========================
    # Variável alvo
    # ==========================

    # CPU do próximo instante (t+1)
    vm["cpu_target"] = vm["CPU usage [%]"].shift(-1)

    # ==========================
    # Lags da CPU
    # ==========================

    vm["cpu_lag1"] = vm["CPU usage [%]"].shift(1)
    vm["cpu_lag2"] = vm["CPU usage [%]"].shift(2)
    vm["cpu_lag3"] = vm["CPU usage [%]"].shift(3)

    # ==========================
    # Memória
    # ==========================

    vm["mem_lag1"] = vm["Memory usage [KB]"].shift(1)

    # ==========================
    # Disco
    # ==========================

    vm["disk_read_lag1"] = vm["Disk read throughput [KB/s]"].shift(1)
    vm["disk_write_lag1"] = vm["Disk write throughput [KB/s]"].shift(1)

    # ==========================
    # Rede
    # ==========================

    vm["net_recv_lag1"] = vm["Network received throughput [KB/s]"].shift(1)
    vm["net_sent_lag1"] = vm["Network transmitted throughput [KB/s]"].shift(1)

    # Remove linhas que ficaram com NaN
    vm = vm.dropna()

    # ==========================
    # Variáveis de entrada
    # ==========================

    X = vm[
        [
            "cpu_lag1",
            "cpu_lag2",
            "cpu_lag3",
            "mem_lag1",
            "disk_read_lag1",
            "disk_write_lag1",
            "net_recv_lag1",
            "net_sent_lag1"
        ]
    ]

    # Agora o alvo é o próximo instante
    y = vm["cpu_target"]

    return X, y


# =====================================================
# DIVISÃO TREINO / TESTE
# =====================================================

def dividir_dados(X, y):

    train_size = int(len(X) * 0.8)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    return X_train, X_test, y_train, y_test


# =====================================================
# TREINAMENTO
# =====================================================

def treinar_modelo(modelo, X_train, y_train):

    modelo.fit(X_train, y_train)

    return modelo


# =====================================================
# PREDIÇÃO
# =====================================================

def prever(modelo, X_test):

    return modelo.predict(X_test)


# =====================================================
# AVALIAÇÃO
# =====================================================

def avaliar_modelo(nome, y_test, predicao):

    return {
        "Modelo": nome,
        "MAE": mean_absolute_error(y_test, predicao),
        "RMSE": mean_squared_error(y_test, predicao) ** 0.5,
        "R²": r2_score(y_test, predicao)
    }


# =====================================================
# IMPORTÂNCIA DAS VARIÁVEIS
# =====================================================

def mostrar_importancias(modelo, X):

    if not hasattr(modelo, "feature_importances_"):
        return

    importancia = pd.DataFrame({
        "Variável": X.columns,
        "Importância": modelo.feature_importances_
    })

    importancia = importancia.sort_values(
        by="Importância",
        ascending=False
    )

    print("\nImportância das variáveis")
    print(importancia)


# =====================================================
# PROGRAMA PRINCIPAL
# =====================================================

def main():

    df = carregar_dados()

    X, y = preparar_dados(df)

    X_train, X_test, y_train, y_test = dividir_dados(X, y)

    resultados = []

    # ----------------------------
    # Regressão Linear
    # ----------------------------

    modelo_lr = treinar_modelo(
        LinearRegression(),
        X_train,
        y_train
    )

    pred_lr = prever(modelo_lr, X_test)

    resultados.append(
        avaliar_modelo(
            "Regressão Linear",
            y_test,
            pred_lr
        )
    )

    # ----------------------------
    # Random Forest
    # ----------------------------

    modelo_rf = treinar_modelo(
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        X_train,
        y_train
    )

    pred_rf = prever(modelo_rf, X_test)

    resultados.append(
        avaliar_modelo(
            "Random Forest",
            y_test,
            pred_rf
        )
    )

    comparacao = pd.DataFrame({
        "CPU Real": y_test.values,
        "CPU Prevista": pred_rf
    })

    print(comparacao.head(20))

    resultado = pd.DataFrame(resultados)

    print("\nResultados")
    print(resultado.round(3))

    mostrar_importancias(modelo_rf, X)

    plt.figure(figsize=(15, 6))

    plt.plot(
        y_test.values[:300],
        label="CPU Real"
    )

    plt.plot(
        pred_rf[:300],
        label="CPU Prevista"
    )

    plt.title("Predição da CPU - Random Forest")
    plt.xlabel("Instante de tempo")
    plt.ylabel("CPU (%)")
    plt.legend()

    plt.show()

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_test,
        pred_rf,
        alpha=0.3
    )

    plt.xlabel("CPU Real")
    plt.ylabel("CPU Prevista")
    plt.title("Real x Previsto")

    # Linha ideal (y = x)
    limites = [
        min(y_test.min(), pred_rf.min()),
        max(y_test.max(), pred_rf.max())
    ]

    plt.plot(limites, limites, "r--")

    plt.show()

    for i in range(10):
        print(
            f"Real: {y_test.iloc[i]:6.2f} | Previsto: {pred_rf[i]:6.2f}"
        )


if __name__ == "__main__":
    main()