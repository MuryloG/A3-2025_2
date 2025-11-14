from flask import Flask, request, render_template
import pickle
import numpy as np
import warnings

# Ignorar os avisos do scikit-learn
warnings.filterwarnings('ignore', category=UserWarning)

# Inicializa o aplicativo Flask
app = Flask(__name__)

# --- 1. CARREGAR OS MODELOS ---
try:
    with open('model/modelo_cancer.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler_cancer.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("ERRO: Arquivos 'modelo_cancer.pkl' ou 'scaler_cancer.pkl' não encontrados.")
    print("Execute o script 'rodar_projeto.py' primeiro para treinar e salvar os modelos.")
    exit()

# --- 2. ROTA DA PÁGINA INICIAL ---
@app.route('/')
def home():
    # Apenas renderiza o arquivo HTML
    return render_template('index.html')

# --- 3. ROTA DA PREVISÃO ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pega os dados enviados pelo formulário HTML
        dados_string = request.form['dados_paciente']
        
        # Converte a string (ex: "17.99, 10.38, ...") em uma lista de floats
        dados_lista_str = dados_string.split(',')
        dados_lista_float = [float(i.strip()) for i in dados_lista_str]

        # Verifica se o usuário inseriu os 30 valores
        if len(dados_lista_float) != 30:
            erro_msg = f"Entrada inválida. Foram recebidos {len(dados_lista_float)} valores, mas 30 eram esperados."
            return render_template('index.html', erro=erro_msg)

        # Prepara os dados para o modelo (igual ao prever.py)
        dados_formatados = np.array(dados_lista_float).reshape(1, -1)
        dados_normalizados = scaler.transform(dados_formatados)

        # Faz a previsão
        previsao = model.predict(dados_normalizados)

        # Define o resultado
        if previsao[0] == 1:
            resultado = "MALIGNO"
        else:
            resultado = "BENIGNO"

        # Retorna a mesma página, mas agora com o resultado
        return render_template('index.html', resultado=resultado)

    except Exception as e:
        # Se o usuário digitar letras ou algo quebrar
        erro_msg = f"Erro ao processar os dados. Verifique se os valores estão corretos e separados por vírgula. ({e})"
        return render_template('index.html', erro=erro_msg)


# --- 4. INICIA O SERVIDOR ---
if __name__ == '__main__':
    # 'debug=True' permite que você salve o código e a página atualize sozinha
    app.run(debug=True)