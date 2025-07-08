# === Instalar bibliotecas manualmente no terminal do VS Code ===
# pip install tensorflow scikit-learn matplotlib joblib

# === Importar bibliotecas ===
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib

# === Parâmetros ===
caminho_arquivo_npy = 'data_BK7_Au_5000.npy'
janela = 120
passos_a_frente = 30
batch_size = 32
epochs = 150
taxa_aprendizado = 0.001

# === Carregar e normalizar dados ===
def carregar_dados(scale = 100, size = 2000):
    dados = np.load(caminho_arquivo_npy)
    dados = dados[:size]
    print(f"Tamanho total dos dados: {len(dados)} pontos")

    # Adicionar ruído leve
    # ruido = np.random.normal(0, 0.0005, size=len(dados))
    # dados_ruidosos = np.array(dados) + ruido

    # Normalizar
    scaler = MinMaxScaler(feature_range=(0, scale))


    dados_normalizados = scaler.fit_transform(dados.reshape(-1, 1))
    joblib.dump(scaler, f'scaler_s{scale}_n{size}.gz')

    return dados_normalizados, scaler

# === Preparar dataset ===
def preparar_dataset(dados, janela, passos_frente):
    X, y = [], []
    for i in range(len(dados) - janela - passos_frente + 1):
        X.append(dados[i:i+janela])
        y.append(dados[i+janela:i+janela+passos_frente].flatten())
    return np.array(X), np.array(y)

# === Criar modelo ===
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

# === Plotar resultados ===
def plot_resultados(historico, y_real, y_pred, samples=3):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(historico.history['loss'], label='Train Loss')
    plt.plot(historico.history['val_loss'], label='Validation Loss')
    plt.title('Histórico de Treinamento')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(min(samples, len(y_real))):
        plt.plot(y_real[i], label=f'Real {i+1}')
        plt.plot(y_pred[i], '--', label=f'Predito {i+1}')
    plt.title('Real vs Predito')
    plt.ylabel('Valor Normalizado')
    plt.xlabel('Passos Temporais')
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Pipeline principal ===
def main(scale = 100, size = 2000):
    dados, scaler = carregar_dados(scale, size)
    X, y = preparar_dataset(dados, janela, passos_a_frente)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = criar_modelo_lstm()

    callbacks = [
        EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True),
        ModelCheckpoint(f'melhor_s{scale}_n{size}.h5', save_best_only=True)
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

    print("Avaliando modelo...")
    y_pred = model.predict(X_test)
    plot_resultados(history, y_test, y_pred)

    model.save(f'final_s{scale}_n{size}.h5')
    print("Modelo salvo com sucesso!")

# === Executar ===
if __name__ == "__main__":
    # main(100, 500)
    # main(1, 500)
    # main(100, 2000)
    main(1, 2000)
