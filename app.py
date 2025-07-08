import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tempfile
import os


def calcular_mae(y_original, y_previsao):
    """
    Calcula o Erro Médio Absoluto (MAE).

    Args:
        y_original (np.array): O array de valores reais (observados).
        y_previsao (np.array): O array de valores previstos.

    Returns:
        float: O valor do MAE.
    """
    y_original = np.array(y_original)
    y_previsao = np.array(y_previsao)

    # Calcula a média das diferenças absolutas
    mae = np.mean(np.abs(y_original - y_previsao))

    return mae

def calcular_mse(y_original, y_previsao):
    """
    Calcula o Erro Quadrático Médio (MSE).

    Args:
        y_original (np.array): O array de valores reais (observados).
        y_previsao (np.array): O array de valores previstos.

    Returns:
        float: O valor do MSE.
    """
    y_original = np.array(y_original)
    y_previsao = np.array(y_previsao)

    # Calcula a média das diferenças ao quadrado
    mse = np.mean((y_original - y_previsao) ** 2)

    return mse

def calcular_r2(y_original, y_previsao):
    y_original = np.array(y_original)
    y_previsao = np.array(y_previsao)

    # 1. Calcular a Soma dos Quadrados dos Resíduos (SQR ou SSE)
    sqr = np.sum((y_original - y_previsao) ** 2)

    # 2. Calcular a Soma Total dos Quadrados (STQ ou SST)
    stq = np.sum((y_original - np.mean(y_original)) ** 2)

    # 3. Calcular o R²
    # Adicionamos uma verificação para evitar divisão por zero se todos os valores originais forem iguais
    if stq == 0:
        # Se a variação total for zero, o conceito de R² não se aplica da mesma forma.
        # Se os resíduos também forem zero, a previsão é perfeita (R²=1).
        # Caso contrário, qualquer erro é infinitamente ruim (R²=-inf), mas podemos retornar 0.
        return 1.0 if sqr == 0 else 0.0

    r2 = 1 - (sqr / stq)

    return r2

st.set_page_config(layout="wide")

st.title("Previsão de Sensorgramas com Rede Neural LSTM")

# Parâmetros
start_point = st.number_input("Prever a partir de: ", min_value=120, value=120, max_value=1920)
block_amount = st.number_input("Quantidade de blocos", min_value=1, value=1, max_value=int((1950 - start_point) / 30))

# Arquivos
sensor_data_file = st.file_uploader("Enviar arquivo do sensorgrama (.txt ou .npy)", type=['txt', 'npy'])
model_file = st.file_uploader("Enviar modelo treinado (.h5)", type=['h5'])
scaler_file = st.file_uploader("Enviar scaler (.gz)", type=['gz'])

# Constantes
window = 120
block_size = 30

if st.button("Prever", disabled=not (sensor_data_file and model_file and scaler_file)):
    if sensor_data_file.name.endswith('.txt'):
        data = np.loadtxt(sensor_data_file)
    elif sensor_data_file.name.endswith('.npy'):
        data = np.load(sensor_data_file)
    else:
        data = None
        st.error("⚠️ Apenas arquivos .txt e .npy são suportados.")

    # Carrega os modelos
    model = load_model(model_file.name, compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError())

    # Carrega o scaler
    scaler = joblib.load(scaler_file.name)

    # Aplica o scaler aos dados
    normalized_data = scaler.transform(data.reshape(-1, 1)).flatten()

    st.success("Gerando Previsão...")

    inicio = start_point - window
    fim = inicio + 119

    # === Preparar janela para previsão ===
    entrada = normalized_data[inicio:fim + 1]
    entrada_atual = entrada.copy()
    preditos = []

    # === Previsão sequencial ===
    for _ in range(block_amount):
        entrada_reshape = entrada_atual.reshape(1, window, 1)
        previsao = model.predict(entrada_reshape, verbose=0).flatten()
        preditos.extend(previsao)
        entrada_atual = np.concatenate([entrada_atual[block_size:], previsao])

    # === Desnormalizar previsão ===
    prediction = scaler.inverse_transform(np.array(preditos).reshape(-1, 1)).flatten()

    # === Suavizar previsão ===
    window_length = 11 if len(prediction) >= 11 else len(prediction) | 1  # sempre ímpar
    preditos_suavizado = savgol_filter(prediction, window_length=window_length, polyorder=3)
    final_array = preditos_suavizado

    buffer = data[fim:fim+len(final_array)]
    st.text(f"R²: {r2_score(buffer, final_array):.4f}")
    st.text(f"MAE: {mean_absolute_error(buffer, final_array):.4f}")
    st.text(f"MSE: {mean_squared_error(buffer, final_array):.4f}")

    # Exibição dos gráficos
    st.subheader("📈 Gráfico Comparativo")
    fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
    ax.plot(range(len(data)), data, label="Sensorgrama Original", alpha=0.3)
    # ax.plot(range(start_point, start_point + len(final_array)), buffer, label="Original", color='red', alpha=0.6)
    ax.plot(range(start_point, start_point + len(final_array)), final_array, label="Previsão Suavizada", color='orange')
    ax.set_xlabel("Índice do Ponto")
    ax.set_ylabel("Reflectância")
    ax.set_title("Previsão com LSTM")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

if st.button("Previsão Completa", disabled=not (sensor_data_file and model_file and scaler_file)):
    if sensor_data_file.name.endswith('.txt'):
        data = np.loadtxt(sensor_data_file)
    elif sensor_data_file.name.endswith('.npy'):
        data = np.load(sensor_data_file)
    else:
        data = None
        st.error("⚠️ Apenas arquivos .txt e .npy são suportados.")

    # Carrega os modelos
    model = load_model(model_file.name, compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError())

    # Carrega o scaler
    scaler = joblib.load(scaler_file.name)

    # Aplica o scaler aos dados
    normalized_data = scaler.transform(data.reshape(-1, 1)).flatten()
    final_array = []

    st.success("Gerando Previsão...")

    for i in np.arange(0, len(data) - window - (block_amount * block_size), block_amount * block_size):
        inicio = i
        fim = i + 119

        janela = fim - inicio + 1
        print(f"Janela selecionada de {inicio} até {fim} (tamanho {janela})")

        # === Preparar janela para previsão ===
        entrada = normalized_data[inicio:fim + 1]
        entrada_atual = entrada.copy()
        preditos = []

        # === Previsão sequencial ===
        for _ in range(block_amount):
            entrada_reshape = entrada_atual.reshape(1, janela, 1)
            previsao = model.predict(entrada_reshape, verbose=0).flatten()
            preditos.extend(previsao)
            entrada_atual = np.concatenate([entrada_atual[block_size:], previsao])

        # === Desnormalizar previsão ===
        prediction = scaler.inverse_transform(np.array(preditos).reshape(-1, 1)).flatten()

        # === Suavizar previsão ===
        # window_length = 11 if len(prediction) >= 11 else len(prediction) | 1  # sempre ímpar
        # prediction = savgol_filter(prediction, window_length=window_length, polyorder=3)
        final_array = np.concatenate((final_array, prediction))

    st.text(f"R²: {r2_score(data[119:119 + len(final_array)], final_array):.4f}")
    st.text(f"MAE: {mean_absolute_error(data[119:119 + len(final_array)], final_array):.4f}")
    st.text(f"MSE: {mean_squared_error(data[119:119 + len(final_array)], final_array):.4f}")

    # Exibição dos gráficos
    st.subheader("📈 Gráfico Comparativo")
    fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
    ax.plot(range(len(data)), data, label="Sensorgrama Original", alpha=0.3)
    ax.plot(range(start_point, start_point + len(final_array)), final_array, label="Previsão Suavizada", color='orange')
    ax.set_xlabel("Índice do Ponto")
    ax.set_ylabel("Reflectância")
    ax.set_title("Previsão com LSTM")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
