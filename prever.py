import pickle
import numpy as np

try:
    with open('modelo_cancer.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_cancer.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("ERRO: Arquivos 'modelo_cancer.pkl' ou 'scaler_cancer.pkl' não encontrados.")
    print("Execute o script 'rodar_projeto.py' primeiro para treinar e salvar o modelo.")
    exit()

dados_paciente = [
    12.45, 15.7, 82.57, 477.1, 0.1278, 0.17, 0.1578, 0.08089, 0.2087, 0.07613, 
    0.3345, 0.8902, 2.217, 27.19, 0.00751, 0.03345, 0.03672, 0.01137, 0.02165, 0.005082, 
    15.47, 23.75, 103.4, 741.6, 0.1791, 0.5249, 0.5355, 0.1741, 0.3985, 0.1244
]

dados_formatados = np.array(dados_paciente).reshape(1, -1)

dados_normalizados = scaler.transform(dados_formatados)

previsao = model.predict(dados_normalizados)

print("\n--- Diagnóstico do Protótipo de IA ---")
if previsao[0] == 1:
    print("Resultado: [ MALIGNO ]")
else:
    print("Resultado: [ BENIGNO ]")

print("-----------------------------------------")