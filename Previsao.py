
!pip install tensorflow scikit-learn matplotlib scipy

# 🔧 Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from google.colab import files
import os

# Parâmetros configuráveis
JANELA = 300  # Tamanho da janela de observação
COR_DADOS = '#1f77b4'  # Cor para dados reais
COR_PREVISAO = '#ff7f0e'  # Cor para previsão
COR_CONFIANCA =  '#ffcc00'  # Cor para área de confiança

# Função para upload seguro de arquivos
def upload_arquivo(mensagem, extensoes):
    print(mensagem)
    uploaded = files.upload()
    while not any(f.endswith(tuple(extensoes)) for f in uploaded.keys()):
        print(f"Por favor, envie um arquivo {extensoes}")
        uploaded = files.upload()
    nome_arquivo = next(f for f in uploaded.keys() if any(f.endswith(ext) for ext in extensoes))
    with open(nome_arquivo, 'wb') as f:
        f.write(uploaded[nome_arquivo])
    return nome_arquivo

# Carregar modelo com tratamento de erro
def carregar_modelo_seguro(caminho):
    try:
        return load_model(caminho, compile=False)
    except:
        return load_model(caminho, compile=False,
                        custom_objects={'mse': MeanSquaredError(),
                                      'mean_squared_error': MeanSquaredError()})

#Carregar dados
print("\n" + "="*50)
print("CARREGAMENTO DE ARQUIVOS")
print("="*50)
nome_txt = upload_arquivo("Envie o arquivo .txt do sensorgrama", [".txt"])
dados_brutos = np.loadtxt(nome_txt).reshape(-1, 1)

# Pré-processamento dos dados
dados = savgol_filter(dados_brutos.flatten(), window_length=51, polyorder=3).reshape(-1, 1)

# Carregar modelo e scaler
nome_modelo = upload_arquivo("Envie o modelo (.h5 ou .keras)", [".h5", ".keras"])
model = carregar_modelo_seguro(nome_modelo)

# Verificar arquitetura do modelo
print("\nInformações do Modelo:")
model.summary()

nome_scaler = upload_arquivo("📥 Envie o arquivo .npy do scaler", [".npy"])
scaler_params = np.load(nome_scaler, allow_pickle=True)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = 0, 1/(scaler_params[1] - scaler_params[0])
scaler.data_min_, scaler.data_max_ = scaler_params

#Seleção de intervalo
print("\n" + "="*50)
print("SELEÇÃO DE INTERVALO")
print("="*50)
while True:
    try:
        print(f"\nℹDados disponíveis: {len(dados)} pontos (0-{len(dados)-1})")
        inicio = int(input("Início do intervalo (recomendado: últimos 500 pontos): "))
        fim = int(input(" Fim do intervalo (deve ser igual ao tamanho dos dados para usar todos): "))

        if not (0 <= inicio < fim <= len(dados)):
            print(f"Intervalo inválido. Use valores entre 0-{len(dados)-1}")
            continue

        if (fim - inicio) < JANELA:
            print(f"O intervalo precisa ter pelo menos {JANELA} pontos")
            continue

        break
    except ValueError:
        print("Por favor, insira números válidos")

# Selecionar número de pontos para previsão
print("\n" + "="*50)
print("CONFIGURAÇÃO DA PREVISÃO")
print("="*50)
while True:
    try:
        PASSOS_A_FRENTE = int(input(f"\nQuantos pontos de previsão deseja? (30-500 recomendado): "))
        if 1 <= PASSOS_A_FRENTE <= 1000:
            break
        print("Por favor, insira um número entre 1 e 1000")
    except ValueError:
        print("Por favor, insira um número válido")

# Pré-processamento e verificação
print("\n" + "="*50)
print("PRÉ-PROCESSAMENTO")
print("="*50)

# 1. Selecionar e visualizar janela de entrada
entrada = dados[inicio:fim]
plt.figure(figsize=(12, 4))
plt.plot(entrada, label='Janela de Entrada (filtrada)')
plt.title("Últimos {} Pontos Usados para Previsão".format(len(entrada)))
plt.xlabel("Pontos Relativos")
plt.ylabel("Reflectância")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# 2. Verificar estatísticas
print("\n📊 Estatísticas dos Dados Atuais:")
print("Média:", np.mean(entrada))
print("Desvio Padrão:", np.std(entrada))
print("Máximo:", np.max(entrada))
print("Mínimo:", np.min(entrada))

print("\n📊 Estatísticas dos Dados de Treino (scaler):")
print("Range esperado:", scaler.data_min_, "a", scaler.data_max_)

#Fazer previsão com controle de qualidade
print("\n" + "="*50)
print("REALIZANDO PREVISÃO")
print("="*50)

entrada_norm = scaler.transform(entrada[-JANELA:].reshape(-1, 1)).reshape(1, JANELA, 1)

previsoes = []
confianca = []
max_variation = 0.005 * (scaler.data_max_ - scaler.data_min_)  # Máxima variação permitida

for i in range(PASSOS_A_FRENTE):
    saida_temp = model.predict(entrada_norm, verbose=0)
    
    # Controle de variação
    if i > 0 and abs(saida_temp[0,0] - previsoes[-1]) > max_variation:
        saida_temp[0,0] = previsoes[-1] + np.sign(saida_temp[0,0] - previsoes[-1]) * max_variation
    
    previsoes.append(float(saida_temp[0,0]))  # Garantindo que é um valor float
    
    # Cálculo de confiança (baseado na variação)
    current_confidence = 1 - min(abs(saida_temp[0,0] - previsoes[-2])/max_variation, 1) if i > 0 else 1.0
    confianca.append(float(current_confidence))
    
    # Atualizar entrada para próximo passo
    entrada_norm = np.roll(entrada_norm, -1, axis=1)
    entrada_norm[0, -1, 0] = saida_temp[0,0]
    
    # Progresso
    if (i+1) % 50 == 0 or (i+1) == PASSOS_A_FRENTE:
        print(f"🔹 Progresso: {i+1}/{PASSOS_A_FRENTE} pontos previstos")

# Converter para arrays numpy com formatos consistentes
previsoes = np.array(previsoes).reshape(-1, 1)
saida = scaler.inverse_transform(previsoes).flatten()
confianca = np.array(confianca)

# Visualização completa
print("\n" + "="*50)
print("VISUALIZAÇÃO DOS RESULTADOS")
print("="*50)

plt.figure(figsize=(16, 8))

# Plotar dados históricos completos
plt.plot(np.arange(len(dados)), dados, label="Dados Reais", color=COR_DADOS, alpha=0.7)

# Plotar previsão
plt.plot(np.arange(len(dados), len(dados)+PASSOS_A_FRENTE), saida,
        label=f"Previsão ({PASSOS_A_FRENTE} pontos)", color=COR_PREVISAO, linewidth=2.5)

# Área de confiança
plt.fill_between(np.arange(len(dados), len(dados)+PASSOS_A_FRENTE),
                saida * (1 - confianca*0.05),
                saida * (1 + confianca*0.05),
                color=COR_CONFIANCA, alpha=0.3, label="Confiança (95%)")

# Linha divisória
plt.axvline(x=len(dados)-1, color='k', linestyle='--', alpha=0.5)

plt.title(f"Previsão SPR - {os.path.splitext(nome_txt)[0]} | {PASSOS_A_FRENTE} pontos", pad=20, fontsize=14)
plt.xlabel("Pontos de Medição", fontsize=12)
plt.ylabel("Reflectância", fontsize=12)
plt.grid(True, alpha=0.2)
plt.legend(fontsize=12)
plt.tight_layout()

#Salvar resultados CORRIGIDO
nome_base = os.path.splitext(nome_txt)[0]
output_filename = f"previsao_melhorada_{PASSOS_A_FRENTE}pts_{nome_base}"

# Salvar gráfico
plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')

# Salvar dados em formato correto - CORREÇÃO PRINCIPAL AQUI
indices = np.arange(len(dados), len(dados)+PASSOS_A_FRENTE)
dados_salvar = np.column_stack((indices, saida, confianca))
np.savetxt(f"{output_filename}.csv", dados_salvar,
           delimiter=",", header="indice,valor,confianca", comments='')

print("\nPrevisão concluída com sucesso!")
print(f"Gráfico salvo em: {output_filename}.png")
print(f"Dados da previsão em: {output_filename}.csv")

plt.show()

