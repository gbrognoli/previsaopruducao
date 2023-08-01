import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Dados históricos (Exemplo)
dados_historicos = [
    [2010, 1000, 500, 600],
    [2011, 1200, 600, 550],
    [2012, 800, 400, 700],
    [2013, 1500, 800, 900],
    # Adicione mais dados históricos aqui
]

# Convertendo os dados em arrays NumPy
dados_historicos = np.array(dados_historicos)

# Separando as características (X) e os rótulos (y)
X = dados_historicos[:, 0].reshape(-1, 1)  # Ano
y = dados_historicos[:, 1:]  # Produção de algodão, outros dados históricos, etc.

# Normalizando os dados entre 0 e 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo da rede neural
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4)  # Camada de saída sem função de ativação (regressão)
])

# Compilando o modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Treinando o modelo
modelo.fit(X_train, y_train, epochs=100, batch_size=32)

# Avaliando o modelo no conjunto de teste
loss = modelo.evaluate(X_test, y_test)
print(f'Loss (erro): {loss}')

# Realizando previsões para um novo dado (exemplo: ano 2024)
novo_dado = np.array([[2024]])
novo_dado = scaler.transform(novo_dado)
previsao = modelo.predict(novo_dado)
print(f'Previsão para 2024: {previsao[0]}')
