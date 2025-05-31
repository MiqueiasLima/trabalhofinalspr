import matplotlib.pyplot as plt
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


# 1. Carregar os dados do arquivo
with open(r'C:\Users\mique\OneDrive\Área de Trabalho\Ufersa - Mestrado\Sensores - BioSensores\Aulas\Trabalho Final\dados_BKAg.txt', 'r') as file:
    dados = [float(linha.strip()) for linha in file]

print(f'Total de dados lidos: {len(dados)}')
print(f'Primeiros valores: {dados[:10]}')

dados = np.array(dados).reshape(-1, 1)

# 2. Normalizar os dados
scaler = MinMaxScaler()
dados_norm = scaler.fit_transform(dados)

# 3. Criar janelas de sequência (para previsão)
janela = 100  # número de passos de entrada
X, y = [], []
for i in range(len(dados_norm) - janela):
    X.append(dados_norm[i:i+janela])
    y.append(dados_norm[i+janela])

X = np.array(X)
y = np.array(y)

# 4. Construir a RNN com LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(janela, 1)))
model.add(Dense(1))  # previsão de 1 valor

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=32)

# 5. Fazer previsões
y_pred = model.predict(X)

# 6. Reverter a normalização para comparar com os dados originais
y_true = scaler.inverse_transform(y)
y_pred_inv = scaler.inverse_transform(y_pred)

# 7. Visualizar
plt.figure(figsize=(12, 5))
plt.plot(y_true, label='Real')
plt.plot(y_pred_inv, label='Previsto', alpha=0.7)
plt.legend()
plt.title('Previsão do Sinal SPR com LSTM')
plt.xlabel('Amostras')
plt.ylabel('Reflectância')
plt.grid(True)
plt.show()