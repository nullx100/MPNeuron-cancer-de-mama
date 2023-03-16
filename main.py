from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

## Visualizar datos

dir(breast_cancer)

import pandas as pd 

df = pd.DataFrame(X, columns=breast_cancer.feature_names)
df

Y

## Dividir el conjunto de datos 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, Y, stratify=Y)

print("Tamano del conjunto de datos de entrenamiento: ", len(X_train))
print("Tamano del conjunto de datos de prueba: ", len(X_test))

import numpy as np
from sklearn.metrics import accuracy_score

class MPNeuron:
    def __init__(self):
        self.threshold = None
    
    # Funcion de activacion
    def model(self, x):
        # input: [1, 0, 1, 0] [x1, x2, ..., xn]
        return (sum(x) >= self.threshold)

    def predict(self, X):
        # input: [[1, 0, 1, 0], [1, 0, 1, 1]]
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    
    def fit(self, X, Y):
        accuracy = {}
        # Seleccionamos un threshold entre el # de caracteristicas de entrada
        for th in range(X.shape[1] + 1):
            self.threshold = th 
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y_pred, Y)
        # Seleccionamos el threshold que mejores resultados proporciona
        self.threshod = max(accuracy, key=accuracy.get)

# Ejemplo: Transformar valores a binario
import matplotlib.pyplot as plt

print(pd.cut([0.04, 2, 4, 5, 6, 0.02, 0.6], bins=2, labels=[0,1]))

plt.hist([0.04, 2, 4, 5, 6, 0.02, 0.6], bins=2)
plt.show()

# Transformamos las caracteristicas de entrada a un valor binario
X_train_bin = X_train.apply(pd.cut, bins=2, labels=[1,0])
X_test_bin = X_test.apply(pd.cut, bins=2, labels=[1,0])

X_train_bin

#Instanciamos el modelo MPNeuron
mp_neuron = MPNeuron()

#Encontramos el threshold optimo
mp_neuron.fit(X_train_bin.to_numpy(), y_train)
mp_neuron.threshold

# Realizamos preddiciones para ejemplos nuevos que no se encuentran en el conjunto de datos de entrenamiento
Y_pred = mp_neuron.predict(X_test_bin.to_numpy())
Y_pred

# Calculamos la exactitud de nuestra prediccion
accuracy_score(y_test, Y_pred)


# Calculamos la matriz de confusion
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, Y_pred)
