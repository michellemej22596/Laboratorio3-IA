import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Cargar el dataset Iris
iris = load_iris()

# Seleccionar solo las características que necesitamos (sepal length y sepal width)
X = iris.data[:, :2]  # Todas las filas y solo las dos primeras columnas (sepal length, sepal width)
y = iris.target

# Filtrar para clasificación binaria (solo las dos primeras clases)
X = X[y != 2]
y = y[y != 2]

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Contar la cantidad de muestras en cada clase
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))

print("Distribución de las clases:", class_distribution)

# Inicializar el perceptrón y entrenarlo
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = perceptron.predict(X_test)

# Evaluar el desempeño utilizando la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualizar la frontera de decisión
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))

# Predecir para cada punto del grid
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar la frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', marker='o')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Frontera de Decisión del Perceptrón')
plt.show()
