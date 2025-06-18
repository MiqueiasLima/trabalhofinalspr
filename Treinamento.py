import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import joblib
from tkinter import Tk, filedialog

# 🔧 Parâmetros
janela = 120               # Tamanho da janela temporal
passos_a_frente = 30       # Previsão de 30 pontos futuros
batch_size = 32
epochs = 150
taxa_aprendizado = 0.001

# Função para selecionar arquivo ZIP usando janela de diálogo
def selecionar_zip():
    Tk().withdraw()  # Oculta a janela principal do Tkinter
    print("Selecione o arquivo .zip com os arquivos .txt dos sensorgramas de treino")
    arquivo_zip = filedialog.askopenfilename(title="Selecione o arquivo .zip", filetypes=[("Zip files", "*.zip")])
    if not arquivo_zip:
        raise Exception("Nenhum arquivo selecionado. Finalizando execução.")
    return arquivo_zip

# Extrair arquivos .txt
def extrair_arquivos(arquivo_zip, pasta_destino='sensorgramas'):
    os.makedirs(pasta_destino, exist_ok=True)
    with zipfile.ZipFile(arquivo_zip, 'r') as zip_ref:
        zip_ref.extractall(pasta_destino)
    arquivos = [f for f in os.listdir(pasta_destino) if f.endswith('.txt')]
    print("Arquivos encontrados:", arquivos)
    return arquivos, pasta_destino

# Função para carregar e normalizar dados
def carregar_dados(arquivos, caminho_dados):
    dados_completos = []
    for arquivo in arquivos:
        caminho = os.path.join(caminho_dados, arquivo)
        dados = np.loadtxt(caminho)
        dados_completos.extend(dados)

    # Adicionar ruído leve
    ruido = np.random.normal(0, 0.0005, size=len(dados_completos))
    dados_completos = np.array(dados_completos) + ruido

    # Normalização
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados_completos.reshape(-1, 1))

    return dados_normalizados, scaler

# Preparar dataset para LSTM
def preparar_dataset(dados, janela, passos_frente):
    X, y = [], []
    for i in range(len(dados) - janela - passos_frente + 1):
        X.append(dados[i:i+janela])
        y.append(dados[i+janela:i+janela+passos_frente].flatten())
    return np.array(X), np.array(y)

# Criar modelo LSTM
def criar_modelo_lstm():
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(janela, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(passos_a_frente)
    ])
    optimizer = Adam(learning_rate=taxa_aprendizado)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Visualizar resultados (Corrigido com inverse_transform)
def plot_resultados(historico, y_real, y_pred, scaler, samples=3):
    plt.figure(figsize=(15, 10))

    # Curva de perda
    plt.subplot(2, 1, 1)
    plt.plot(historico.history['loss'], label='Train Loss')
    plt.plot(historico.history['val_loss'], label='Validation Loss')
    plt.title('Histórico de Treinamento')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Inverter normalização para o Y real e o Y predito
    y_real_inv = scaler.inverse_transform(y_real.reshape(-1, 1)).reshape(y_real.shape)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

    # Previsões vs reais (já na escala original)
    plt.subplot(2, 1, 2)
    for i in range(min(samples, len(y_real))):
        plt.plot(y_real_inv[i], label=f'Real {i+1}')
        plt.plot(y_pred_inv[i], '--', label=f'Predito {i+1}')
    plt.title('Real vs Predito (Escala Original)')
    plt.ylabel('Resposta SPR')
    plt.xlabel('Passos Temporais')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Pipeline principal
def main():
    # 1. Selecionar e extrair dados
    arquivo_zip = selecionar_zip()
    arquivos, caminho_dados = extrair_arquivos(arquivo_zip)

    # 2. Carregar dados
    dados, scaler = carregar_dados(arquivos, caminho_dados)
    X, y = preparar_dataset(dados, janela, passos_a_frente)

    # 3. Treino/teste
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 4. Criar e treinar modelo
    model = criar_modelo_lstm()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint('melhor_modelo.h5', save_best_only=True)
    ]

    print("Iniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Avaliar
    print("Avaliando modelo...")
    y_pred = model.predict(X_test)
    plot_resultados(history, y_test, y_pred, scaler)

    # 6. Salvar modelo e scaler
    model.save('modelo_spr_final.h5')
    joblib.dump(scaler, 'scaler_spr_final.gz')
    print("Modelo e scaler salvos com sucesso!")

# Rodar pipeline
if __name__ == "__main__":
    main()
