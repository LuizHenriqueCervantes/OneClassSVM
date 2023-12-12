import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

#Carregar dados do arquivo excel
excel_path = r'C:\Users\hrick\Documents\VsCode\Python\IA\anomalias\AnomaliasTransacoesBancarias.xlsx'

df = pd.read_excel(excel_path)

#Plotar dados originais
plt.figure(figsize=(12, 6))
plt.plot(df['Data'], df['Valores'], marker='o', linestyle='-')
plt.title('Gráfico original')
plt.xlabel('Data')
plt.ylabel('Valor(R$)')
plt.show()

#Utilizar a coluna 'Valores' como dados a serem analisados
y = df['Valores'].values.reshape(-1, 1)

#Treinar o modelo One-Class SVM
model = OneClassSVM(nu=0.1)
model.fit(y)

#Prever anomalias
predictions = model.predict(y)

#Plotar e destacar as anomalias
plt.figure(figsize=(12, 6))
plt.plot(df['Data'], df['Valores'], marker='o', linestyle='-', label='Dados')
plt.scatter(df['Data'][predictions == -1], df['Valores'][predictions == -1], 
            color='red', label='Anomalias', s=100, marker='X')
plt.title('Detecção de Anomalias (One-Class SVM)')
plt.xlabel('Data')
plt.ylabel('Valor (R$)')
plt.legend()
plt.show()
