import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def calcular_media_desvio(resultados):
    return np.mean(resultados), np.std(resultados)

def carregar_dados():
    dados = np.load('teste1.npy')
    return dados[0], np.ravel(dados[1])

def treinar_avaliar_mlp(x, y):
    regr = MLPRegressor(hidden_layer_sizes=(10),
                        max_iter=1000,
                        activation='relu',
                        solver='adam',
                        learning_rate='adaptive',
                        n_iter_no_change=50)
    
    regr.fit(x, y)
    y_pred = regr.predict(x)
    erro = np.mean((y_pred - y) ** 2)
    
    plt.figure(figsize=[14, 7])
    plt.subplot(1, 3, 1)
    plt.plot(x, y)
    plt.subplot(1, 3, 2)
    plt.plot(regr.loss_curve_)
    plt.subplot(1, 3, 3)
    plt.plot(x, y, linewidth=1, color='red')
    plt.plot(x, y_pred, linewidth=2)
    plt.show()
    
    return erro

def main():
    resultados_erro = [treinar_avaliar_mlp(*carregar_dados()) for _ in range(10)]
    media_erro, desvio_padrao_erro = calcular_media_desvio(resultados_erro)
    print("Média do erro:", media_erro)
    print("Desvio padrão do erro:", desvio_padrao_erro)

if __name__ == "__main__":
    main()
