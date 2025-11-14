# Projeto A3: Protótipo de IA em Saúde (2025.2)

Este repositório contém o "Produto Final" para a UC de Inteligência Artificial, um protótipo funcional de um modelo de IA focado na prevenção de doenças, especificamente o diagnóstico de Câncer de Mama.

O modelo é uma **Rede Neural (MLPClassifier)** treinada com o dataset público "Wisconsin Breast Cancer" para classificar tumores como **Malignos** ou **Benignos** com base em 30 características laboratoriais.

## 🚀 Protótipo Funcional

O projeto é dividido em dois scripts principais:
1.  `rodar_projeto.py`: Script completo que carrega o dataset, pré-processa os dados, treina a Rede Neural e salva o modelo final (`modelo_cancer.pkl`) e o relatório de métricas (`relatorio_metricas.txt`).
2.  `prever.py`: Um script de demonstração que carrega o modelo treinado e o utiliza para fazer uma previsão em um novo "paciente", simulando um caso de uso real.

## 📊 Resultados do Modelo

O protótipo atingiu um desempenho de alta confiabilidade, validando sua eficácia como ferramenta de apoio ao diagnóstico.

* **Acurácia Global:** 98.25%
* **Recall (Maligno):** 95% (O modelo identificou corretamente 95% de todos os casos malignos reais)
* **Precision (Maligno):** 100% (Quando o modelo previu "Maligno", ele estava 100% correto)

*Resultados completos estão disponíveis em `relatorio_metricas.txt`.*

## ⚙️ Como Executar o Projeto

Para executar este protótipo em um novo computador, siga os passos abaixo.

### 1. Pré-requisitos

* [Python 3.10+](https://www.python.org/downloads/)
* O dataset `data.csv` (incluído neste repositório)

### 2. Instalação

Clone o repositório e instale as dependências dentro de um ambiente virtual (`.venv`):

```bash
# Clone este repositório
git clone [URL_DO_SEU_REPOSITORIO]
cd [NOME_DA_PASTA_DO_PROJETO]

# Crie e ative o ambiente virtual
python -m venv .venv
.\.venv\Scripts\activate

# Instale as bibliotecas necessárias
pip install pandas scikit-learn
