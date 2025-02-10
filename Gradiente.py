import numpy as np
import matplotlib.pyplot as plt

import numpy as np

# Coeficientes del polinomio: 2x^3 - 3x^2 + 5x + 3
coefficients = [2, -3, 5, 3]

# Función polinómica utilizando np.polyval
def f(x):
    return np.polyval(coefficients, x)

# Derivada de la función polinómica utilizando np.polyder
def df(x):
    derivative_coeff = np.polyder(coefficients)  # Derivada del polinomio
    return np.polyval(derivative_coeff, x)


def gradient_descent(learning_rate, n_iterations, initial_guess):
    x = initial_guess
    history = [x]  # Para almacenar el progreso
    for _ in range(n_iterations):
        grad = df(x)
        x = x - learning_rate * grad
        history.append(x)
    return np.array(history)

def stochastic_gradient_descent(learning_rate, n_iterations, initial_guess, batch_size=1):
    x = initial_guess
    history = [x]
    for _ in range(n_iterations):
        # Seleccionar un batch aleatorio de tamaño 1 (mini-batch)
        x_batch = np.random.uniform(-10, 10, batch_size)  # Cambiar el rango si es necesario
        grad_batch = np.mean([df(xi) for xi in x_batch])  # Promedio del gradiente
        x = x - learning_rate * grad_batch
        history.append(x)
    return np.array(history)

# Parámetros
learning_rate = 0.001
n_iterations = 100
initial_guess = 0  # Puedes cambiar el valor inicial

# Ejecutar los algoritmos
gd_history = gradient_descent(learning_rate, n_iterations, initial_guess)
sgd_history = stochastic_gradient_descent(learning_rate, n_iterations, initial_guess)

# Graficar la función y las aproximaciones
x_values = np.linspace(-10, 10, 400)
y_values = f(x_values)

plt.plot(x_values, y_values, label='Función Polinómica', color='blue')
plt.plot(gd_history, f(gd_history), label='Descenso de Gradiente', color='green')
plt.plot(sgd_history, f(sgd_history), label='SGD', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparación entre Descenso de Gradiente y SGD')
plt.show()

import time

# Medir el tiempo de ejecución de ambos métodos
start_time = time.time()
gd_history = gradient_descent(learning_rate, n_iterations, initial_guess)
gd_time = time.time() - start_time

start_time = time.time()
sgd_history = stochastic_gradient_descent(learning_rate, n_iterations, initial_guess)
sgd_time = time.time() - start_time

print(f"Tiempo de ejecución GD: {gd_time:.4f} segundos")
print(f"Tiempo de ejecución SGD: {sgd_time:.4f} segundos")

# Comparar el fitness (ajuste final)
gd_fitness = f(gd_history[-1])
sgd_fitness = f(sgd_history[-1])

print(f"Fitness GD: {gd_fitness:.4f}")
print(f"Fitness SGD: {sgd_fitness:.4f}")
