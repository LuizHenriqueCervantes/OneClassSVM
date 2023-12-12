import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

#Carregar dados a partir de um arquivo Excel
excel_path = r'C:\Users\hrick\Documents\VsCode\Python\IA\anomalias\TemperaturaPoucaAnomalia.xlsx'

df = pd.read_excel(excel_path)

#Converter a coluna 'Hora' para o formato string
df['Hora'] = df['Hora'].astype(str)

#Utilizar a coluna 'Graus' como dados a serem analisados
y = df['Graus'].values.reshape(-1, 1)

#Treinar o modelo One-Class SVM
model = OneClassSVM(nu=0.3)
model.fit(y)

#Prever anomalias
predictions = model.predict(y)

#Remover anomalias e calcular a média dos Graus não anômalos
valores_sem_anomalias = df.loc[predictions == 1, 'Graus']
media_sem_anomalias = valores_sem_anomalias.mean()

#Imprimir a média dos Graus não anômalos
print(f"Média dos Graus sem anomalias: {media_sem_anomalias}")

#Plotar os dados com as anomalias
plt.figure(figsize=(12, 6))
plt.plot(df['Hora'], df['Graus'], marker='o', linestyle='-', label='Dados')
plt.scatter(df['Hora'][predictions == -1], df['Graus'][predictions == -1],
            color='red', label='Anomalias', s=100, marker='X')
plt.title('Detecção de Anomalias (One-Class SVM)')
plt.xlabel('Hora')
plt.ylabel('Graus')
plt.xticks(df['Hora'][::2], rotation=45)
plt.legend()
plt.show()
