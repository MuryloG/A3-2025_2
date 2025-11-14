import pandas as pd
import sys
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

NOME_DATASET = 'data.csv' 
ARQUIVO_RELATORIO_FINAL = 'relatorio_metricas.txt'

def main():
    print("--- INICIANDO SCRIPT DE TREINAMENTO ---")
    
    print(f"\n[ETAPA 2] Carregando e preparando '{NOME_DATASET}'...")
    
    try:
        df = pd.read_csv(NOME_DATASET)
    except FileNotFoundError:
        print(f"\nERRO FATAL: Arquivo '{NOME_DATASET}' não encontrado.")
        print("Por favor, coloque o 'data.csv' na mesma pasta do script.")
        sys.exit(1) 

    colunas_para_remover = ['id', 'Unnamed: 32']
    for col in colunas_para_remover:
        if col in df.columns:
            df = df.drop(col, axis=1)
            
    df = df.fillna(0)
    print("Dados carregados e limpos.")

    X = df.drop('diagnosis', axis=1)
    y_text = df['diagnosis']

    print("Codificando alvo para classificação binária...")
    y = y_text.apply(lambda x: 1 if x == 'M' else 0)
    class_names = ['Benigno', 'Maligno']
    print(f"Classes: 2 ({class_names[0]}, {class_names[1]})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2,    
                                                        random_state=42,  
                                                        stratify=y) 
    
    print("Normalizando os dados (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Dados pré-processados e prontos para o treino.")
    
    print("\n[ETAPA 3] Iniciando treinamento da Rede Neural (MLPClassifier)...")
    print("(Isso pode levar alguns segundos...)")

    model = MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        max_iter=1000,                
        activation='relu',            
        solver='adam',                
        random_state=42,
        verbose=False                 
    )
    
    model.fit(X_train_scaled, y_train)
    print("Modelo treinado com sucesso!")

    print("\n[ETAPA 4] Avaliando desempenho do modelo...")
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    
    print("Avaliação concluída.")
    
    print(f"\n--- AVALIAÇÃO FINAL DO PROTÓTIPO ---")
    print(f"Acurácia Global: {accuracy * 100:.2f}%")
    print("\nRelatório de Classificação (Detalhado):")
    print(report)

    try:
        with open(ARQUIVO_RELATORIO_FINAL, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE AVALIAÇÃO DO PROTÓTIPO DE IA\n")
            f.write("="*50 + "\n\n")
            f.write(f"Algoritmo: MLPClassifier (Rede Neural)\n")
            f.write(f"Dataset: {NOME_DATASET} (Wisconsin Breast Cancer)\n")
            f.write(f"Operação: Classificação Binária (Maligno vs. Benigno)\n\n")
            f.write(f"ACURÁCIA GLOBAL: {accuracy * 100:.2f}%\n\n")
            f.write("--- MÉTRICAS DE CLASSIFICAÇÃO ---\n")
            f.write(report)
            
        print(f"\n[SUCESSO] Resultados salvos em '{ARQUIVO_RELATORIO_FINAL}'")
        
    except Exception as e:
        print(f"\n[ERRO] Não foi possível salvar o arquivo de relatório: {e}")
        
    
    print("\n[ETAPA 6] Salvando modelo e scaler para produção...")
    with open('modelo_cancer.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler_cancer.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Modelo 'modelo_cancer.pkl' e 'scaler_cancer.pkl' salvos.")

    print("\n--- SCRIPT CONCLUÍDO ---")

if __name__ == "__main__":
    main()