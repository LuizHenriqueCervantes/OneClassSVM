import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

# Criar dados fictícios
np.random.seed(500)
data = {'Valores': np.sin(np.linspace(0, 2 * np.pi, 215)) + 2 * np.random.randn(215)}
df = pd.DataFrame(data)

# Adicionar anomalias
df.loc[(df.index >= 150) & (df.index <= 200), 'Valores'] += 15

# Criar datas fictícias para associar aos valores
df['Data'] = pd.date_range('2023-05-01', '2023-12-01', freq='D')

# Plotar dados originais
plt.figure(figsize=(12, 6))
plt.plot(df['Data'], df['Valores'], marker='o', linestyle='-')
plt.title('Gráfico original')
plt.xlabel('Data')
plt.ylabel('Valores')
plt.show()

# Utilizar a coluna 'Valores' como dados a serem analisados
y = df['Valores'].values.reshape(-1, 1)

# Treinar o modelo One-Class SVM
model = OneClassSVM(nu=0.25)
model.fit(y)

# Prever anomalias
predictions = model.predict(y)

# Plotar e destacar as anomalias
plt.figure(figsize=(12, 6))
plt.plot(df['Data'], df['Valores'], marker='o', linestyle='-', label='Dados')
plt.scatter(df['Data'][predictions == -1], df['Valores'][predictions == -1],
            color='red', label='Anomalias', s=100, marker='x')
plt.title('Detecção de Anomalias (One-Class SVM)')
plt.xlabel('Data')
plt.ylabel('Valores')
plt.legend()
plt.show()