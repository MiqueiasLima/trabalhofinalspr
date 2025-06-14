import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def carregar_sensorgrama(caminho_arquivo, passo_x=1.0):
    """
    Lê o arquivo de valores Y e gera o eixo X com espaçamento definido.
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    y = np.loadtxt(caminho_arquivo)
    x = np.arange(len(y)) * passo_x
    return x, y

def alterar_dados(x, y, delta_x=0.0, delta_y=0.0):
    """
    Altera os valores de X e Y adicionando um delta (positivo ou negativo).
    """
    x_alterado = x + delta_x
    y_alterado = y + delta_y
    return x_alterado, y_alterado

def salvar_y_apenas(y, caminho_saida):
    """
    Salva apenas os valores de Y em um novo arquivo .txt, um por linha.
    """
    np.savetxt(caminho_saida, y, fmt='%.8f')  # Usa 8 casas decimais
    print(f"\nArquivo Y salvo em: {caminho_saida}")

def plotar_sensorgrama(x, y, titulo="Sensorgrama Modificado"):
    """
    Plota o sensorgrama (X vs Y).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Sensorgrama', color='blue')
    plt.title(titulo)
    plt.xlabel('Tempo (X)')
    plt.ylabel('Resposta SPR (Y)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === INTERAÇÃO COM USUÁRIO ===

# Caminho do arquivo .txt de entrada
caminho_entrada = input("Digite o caminho do arquivo .txt com os dados Y: ").strip()

# Definições de parâmetros
passo_x = float(input("Digite o passo para o eixo X (ex: 1.0): ").strip() or "1.0")
delta_x = float(input("Delta X (modificação no eixo X, ex: 0.0): ").strip() or "0.0")
delta_y = float(input("Delta Y (modificação no eixo Y, ex: 0.2): ").strip() or "0.0")

# Definir caminho de saída automaticamente com sufixo
nome_saida = os.path.splitext(os.path.basename(caminho_entrada))[0] + "_Y_modificado.txt"
caminho_saida = os.path.join(os.path.dirname(caminho_entrada), nome_saida)

# Executar o processamento
try:
    x, y = carregar_sensorgrama(caminho_entrada, passo_x)
    x_mod, y_mod = alterar_dados(x, y, delta_x, delta_y)
    salvar_y_apenas(y_mod, caminho_saida)
    plotar_sensorgrama(x_mod, y_mod)
except Exception as e:
    print(f"\nErro: {e}")
    print("Teste")
