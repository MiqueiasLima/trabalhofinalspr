# === Instalar bibliotecas manualmente no terminal do VS Code ===
# pip install tensorflow scikit-learn matplotlib joblib scipy

# === Importar bibliotecas ===
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import joblib
import os

# === Parâmetros de previsão ===
passos_a_frente = 30
n_blocos = 3  # total de blocos para previsão (ex: 2 x 30 = 60 pontos)

# === Caminhos dos arquivos locais ===
caminho_sensorgrama = 'Data/BK7-Au.txt'
caminho_scaler = 'scaler_s100_n2000.gz'
caminho_modelo = 'melhor_s100_n2000.h5'

# === Carregar dados do sensorgrama ===
dados_originais = np.loadtxt(caminho_sensorgrama)
tamanho_total = len(dados_originais)
print(f"Tamanho total do sensorgrama: {tamanho_total} pontos")

final_array = []

for i in np.arange(0, 1950-120-(n_blocos * passos_a_frente), n_blocos * passos_a_frente):
    inicio = i
    fim = i + 119

    janela = fim - inicio + 1
    print(f"Janela selecionada de {inicio} até {fim} (tamanho {janela})")

    # === Carregar scaler ===
    scaler = joblib.load(caminho_scaler)

    # === Normalizar dados ===
    dados_norm = scaler.transform(dados_originais.reshape(-1, 1)).flatten()

    # === Carregar modelo treinado ===
    modelo = load_model(caminho_modelo, compile=False)
    modelo.compile(optimizer='adam', loss=MeanSquaredError())

    # === Preparar janela para previsão ===
    entrada = dados_norm[inicio:fim + 1]
    entrada_atual = entrada.copy()
    preditos = []

    # === Previsão sequencial ===
    for _ in range(n_blocos):
        entrada_reshape = entrada_atual.reshape(1, janela, 1)
        previsao = modelo.predict(entrada_reshape, verbose=0).flatten()
        preditos.extend(previsao)
        entrada_atual = np.concatenate([entrada_atual[passos_a_frente:], previsao])

    # === Desnormalizar previsão ===
    preditos_real = scaler.inverse_transform(np.array(preditos).reshape(-1, 1)).flatten()

    # === Suavizar previsão ===
    window_length = 11 if len(preditos_real) >= 11 else len(preditos_real) | 1  # sempre ímpar
    preditos_suavizado = savgol_filter(preditos_real, window_length=window_length, polyorder=3)
    final_array = np.concatenate((final_array, preditos_suavizado))


# === Plotar resultados ===
plt.figure(figsize=(15, 6))
plt.plot(range(len(dados_originais)), dados_originais, label="Dados Originais Completos", color='green', alpha=0.3)
# plt.plot(range(inicio, fim + 1), dados_originais[inicio:fim + 1], label="Janela de Entrada")

# Adiciona 120 a cada valor de x para deslocar o gráfico
plt.plot(range(120, 120 + len(final_array)), final_array, label=f"Previsão Suavizada ({len(preditos_suavizado)} pontos)", color='orange')

plt.axvline(x=fim, color='gray', linestyle='--')
plt.title(f"Previsão SPR Suavizada - {os.path.basename(caminho_sensorgrama)}")
plt.xlabel("Índice do ponto")
plt.ylabel("Reflectância")
plt.ylim(68, 69.5)  # Define o limite fixo do eixo y entre 68 e 69.5
plt.xlim(-20, 2000)  # Define o limite fixo do eixo x entre -20 e 2000
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# # === Input do intervalo para janela de entrada ===
# while True:
#     try:
#         inicio = int(input(f"Digite o índice inicial (0 a {tamanho_total - 1}): "))
#         fim = int(input(f"Digite o índice final (maior que {inicio} e até {tamanho_total - 1}): "))
#         if 0 <= inicio < fim < tamanho_total:
#             break
#         else:
#             print("Intervalo inválido, tente novamente.")
#     except Exception as e:
#         print("Entrada inválida, digite números inteiros.")

