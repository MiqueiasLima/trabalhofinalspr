

!pip install tensorflow scikit-learn matplotlib

# 🔧 Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from google.colab import files
import zipfile

#Upload do ZIP com arquivos .txt dos sensorgramas
print("Envie um arquivo .zip com os arquivos .txt dos sensorgramas de treino")
uploaded = files.upload()

#Extrair os arquivos .txt
for file in uploaded.keys():
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall('/content/sensorgramas')

# Caminho dos dados
caminho_dados = '/content/sensorgramas'
arquivos = [f for f in os.listdir(caminho_dados) if f.endswith('.txt')]
print("📁 Arquivos encontrados:", arquivos)

#Parâmetros 
janela = 120  # Janela temporal otimizada para SPR
passos_a_frente = 30  # Previsão de curto prazo primeiro
batch_size = 32
epochs = 150
taxa_aprendizado = 0.001

#Função para carregar e normalizar dados
def carregar_dados():
    dados_completos = []
    for arquivo in arquivos:
        caminho = os.path.join(caminho_dados, arquivo)
        dados = np.loadtxt(caminho)
        dados_completos.extend(dados)

    # Adicionar ruído gaussiano leve
    ruido = np.random.normal(0, 0.0005, size=len(dados_completos))
    dados_completos = np.array(dados_completos) + ruido

    # Normalização global
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = scaler.fit_transform(dados_completos.reshape(-1, 1))

    return dados_normalizados, scaler

#Preparar dados para treinamento
def preparar_dataset(dados, janela, passos_frente):
    X, y = [], []
    for i in range(len(dados) - janela - passos_frente + 1):
        X.append(dados[i:i+janela])
        y.append(dados[i+janela:i+janela+passos_frente].flatten())
    return np.array(X), np.array(y)

#Criar modelo LSTM aprimorado
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

#Visualizar resultados
def plot_resultados(historico, y_real, y_pred, samples=3):
    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(historico.history['loss'], label='Train Loss')
    plt.plot(historico.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot predictions
    plt.subplot(2, 1, 2)
    for i in range(min(samples, len(y_real))):
        plt.plot(y_real[i], label=f'Real {i+1}')
        plt.plot(y_pred[i], '--', label=f'Predito {i+1}')
    plt.title('Comparação: Real vs Predito')
    plt.ylabel('Valor Normalizado')
    plt.xlabel('Passos Temporais')
    plt.legend()
    plt.tight_layout()
    plt.show()

#Pipeline principal
def main():
    # 1. Carregar e preparar dados
    dados, scaler = carregar_dados()
    X, y = preparar_dataset(dados, janela, passos_a_frente)

    # 2. Dividir em treino e teste
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 3. Criar e treinar modelo
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

    # 4. Avaliar resultados
    print("Avaliando modelo...")
    y_pred = model.predict(X_test)
    plot_resultados(history, y_test, y_pred)

    # 5. Salvar modelo e scaler
    model.save('modelo_spr_final.h5')
    np.save('scaler_params.npy', np.array([scaler.data_min_, scaler.data_max_]))
    print("Modelo e scaler salvos com sucesso!")

if __name__ == "__main__":
    main()