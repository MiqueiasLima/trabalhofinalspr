import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.losses import MeanSquaredError
from scipy.signal import savgol_filter
import os
import joblib

# Parâmetros configuráveis
JANELA = 300  # Tamanho da janela de observação
COR_DADOS = '#1f77b4'  # Cor para dados reais
COR_PREVISAO = '#ff7f0e'  # Cor para previsão
COR_CONFIANCA = '#ffcc00'  # Cor para área de confiança

# Função para pedir um caminho de arquivo com verificação de extensão
def solicitar_caminho_arquivo(mensagem, extensoes):
    while True:
        caminho = input(f"{mensagem} (Caminho para arquivo com extensão {extensoes}): ").strip()
        if not os.path.isfile(caminho):
            print("❌ Arquivo não encontrado. Tente novamente.")
            continue
        if not any(caminho.lower().endswith(ext) for ext in extensoes):
            print(f"❌ Extensão inválida. Aceito apenas: {extensoes}")
            continue
        return caminho

# Carregar modelo com tratamento de erro
def carregar_modelo_seguro(caminho):
    try:
        return load_model(caminho, compile=False)
    except:
        return load_model(caminho, compile=False,
                          custom_objects={'mse': MeanSquaredError(),
                                          'mean_squared_error': MeanSquaredError()})

def main():
    print("\n" + "=" * 50)
    print("CARREGAMENTO DE ARQUIVOS")
    print("=" * 50)

    # ==== Carregar dados txt ====
    caminho_txt = solicitar_caminho_arquivo("Informe o caminho do arquivo .txt do sensorgrama", [".txt"])
    dados_brutos = np.loadtxt(caminho_txt).reshape(-1, 1)
    dados = savgol_filter(dados_brutos.flatten(), window_length=51, polyorder=3).reshape(-1, 1)

    # ==== Carregar modelo ====
    caminho_modelo = solicitar_caminho_arquivo("Informe o caminho do modelo (.h5 ou .keras)", [".h5", ".keras"])
    model = carregar_modelo_seguro(caminho_modelo)

    print("\nInformações do Modelo:")
    model.summary()

    # ==== Carregar scaler ====
    caminho_scaler = solicitar_caminho_arquivo("Informe o caminho do arquivo .gz do scaler (joblib)", [".gz"])
    scaler = joblib.load(caminho_scaler)

    print("\n" + "=" * 50)
    print("SELEÇÃO DE INTERVALO")
    print("=" * 50)

    while True:
        try:
            print(f"\nℹ Dados disponíveis: {len(dados)} pontos (0 - {len(dados) - 1})")
            inicio = int(input("Início do intervalo (ex: 0): "))
            fim = int(input("Fim do intervalo (ex: 500): "))

            if not (0 <= inicio < fim <= len(dados)):
                print(f"Intervalo inválido. Use valores entre 0 e {len(dados) - 1}")
                continue

            if (fim - inicio) < JANELA:
                print(f"O intervalo precisa ter pelo menos {JANELA} pontos")
                continue

            break
        except ValueError:
            print("Por favor, insira números válidos.")

    print("\n" + "=" * 50)
    print("CONFIGURAÇÃO DA PREVISÃO")
    print("=" * 50)

    while True:
        try:
            PASSOS_A_FRENTE = int(input("\nQuantos pontos de previsão deseja? (30-500 recomendado): "))
            if 1 <= PASSOS_A_FRENTE <= 1000:
                break
            print("Por favor, insira um número entre 1 e 1000.")
        except ValueError:
            print("Por favor, insira um número válido.")

    # =================== PREVISÃO ===================

    entrada = dados[inicio:fim]
    entrada_norm = scaler.transform(entrada[-JANELA:].reshape(-1, 1)).reshape(1, JANELA, 1)

    previsoes = []
    confianca = []
    max_variation = 0.1 * (scaler.data_max_ - scaler.data_min_)

    for i in range(PASSOS_A_FRENTE):
        saida_temp = model.predict(entrada_norm, verbose=0)

        if i > 0 and abs(saida_temp[0, 0] - previsoes[-1]) > max_variation:
            saida_temp[0, 0] = previsoes[-1] + np.sign(saida_temp[0, 0] - previsoes[-1]) * max_variation

        previsoes.append(float(saida_temp[0, 0]))
        conf = 1 - min(abs(saida_temp[0, 0] - previsoes[-2]) / max_variation, 1) if i > 0 else 1.0
        confianca.append(float(conf))

        entrada_norm = np.roll(entrada_norm, -1, axis=1)
        entrada_norm[0, -1, 0] = saida_temp[0, 0]

        if (i + 1) % 50 == 0 or (i + 1) == PASSOS_A_FRENTE:
            print(f"🔹 Progresso: {i + 1}/{PASSOS_A_FRENTE} pontos previstos")

    previsoes = np.array(previsoes).reshape(-1, 1)
    saida = scaler.inverse_transform(previsoes).flatten()
    confianca = np.array(confianca)

    print("\n" + "=" * 50)
    print("VISUALIZAÇÃO DOS RESULTADOS")
    print("=" * 50)

    plt.figure(figsize=(16, 8))

    pontos_janela = np.arange(inicio, fim)
    pontos_prev = np.arange(fim, fim + PASSOS_A_FRENTE)

    plt.plot(pontos_janela, dados[inicio:fim], label="Janela de Entrada", color=COR_DADOS, linewidth=1.5)
    plt.plot(pontos_prev, saida, label=f"Previsão ({PASSOS_A_FRENTE} pontos)", color=COR_PREVISAO, linewidth=2.5)

    plt.fill_between(pontos_prev,
                     saida * (1 - confianca * 0.05),
                     saida * (1 + confianca * 0.05),
                     color=COR_CONFIANCA, alpha=0.3, label="Confiança (95%)")

    plt.axvline(x=fim - 1, color='k', linestyle='--', alpha=0.5)

    nome_base = os.path.splitext(os.path.basename(caminho_txt))[0]
    plt.title(f"Previsão SPR - {nome_base} | {PASSOS_A_FRENTE} pontos", pad=20, fontsize=14)
    plt.xlabel("Pontos de Medição", fontsize=12)
    plt.ylabel("Reflectância", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_filename = f"previsao_{PASSOS_A_FRENTE}pts_{nome_base}"
    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')

    indices = np.arange(fim, fim + PASSOS_A_FRENTE)
    dados_salvar = np.column_stack((indices, saida, confianca))
    np.savetxt(f"{output_filename}.csv", dados_salvar,
               delimiter=",", header="indice,valor,confianca", comments='')

    print("\n✅ Previsão concluída com sucesso!")
    print(f"📊 Gráfico salvo em: {output_filename}.png")
    print(f"📁 Dados da previsão em: {output_filename}.csv")

    plt.show()

if __name__ == "__main__":
    main()
