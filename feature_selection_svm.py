import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset
df = pd.read_csv("high_diamond_ranked_10min.csv")

# Eliminar la columna gameId (no relevante)
df = df.drop(columns=['gameId'])

# Separar características (X) y variable objetivo (y)
X = df.drop(columns=['blueWins'])
y = df['blueWins']

# Normalizar características en el rango [0,1] (para chi-cuadrado)
scaler_minmax = MinMaxScaler()
X_scaled_mm = scaler_minmax.fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled_mm, y, test_size=0.2, random_state=42, stratify=y)

# 1. Variance Threshold (elimina features con varianza muy baja)
var_thresh = VarianceThreshold(threshold=0.01)
X_train_var = var_thresh.fit_transform(X_train)
X_test_var = var_thresh.transform(X_test)

# 2. SelectKBest con chi-cuadrado (selecciona las 15 mejores features)
k_best = SelectKBest(score_func=chi2, k=15)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# 3. SelectFromModel con RandomForest (selecciona features importantes)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
sfm = SelectFromModel(rf_model, threshold="mean", prefit=True)
X_train_sfm = sfm.transform(X_train)
X_test_sfm = sfm.transform(X_test)

# Función para entrenar un modelo SVM y evaluar su precisión
def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Evaluar precisión con cada método
accuracy_original = train_and_evaluate_svm(X_train, X_test, y_train, y_test)
accuracy_var = train_and_evaluate_svm(X_train_var, X_test_var, y_train, y_test)
accuracy_kbest = train_and_evaluate_svm(X_train_kbest, X_test_kbest, y_train, y_test)
accuracy_sfm = train_and_evaluate_svm(X_train_sfm, X_test_sfm, y_train, y_test)

# Almacenar resultados
accuracy_results = {
    "Original (sin selección)": accuracy_original,
    "Variance Threshold": accuracy_var,
    "SelectKBest (chi-cuadrado)": accuracy_kbest,
    "SelectFromModel (RandomForest)": accuracy_sfm
}

# Mostrar resultados en consola
print("Resultados de precisión del modelo SVM con diferentes técnicas de selección de features:")
for method, accuracy in accuracy_results.items():
    print(f"{method}: {accuracy:.4f}")

# Gráfico de comparación
methods = list(accuracy_results.keys())
accuracies = list(accuracy_results.values())

plt.figure(figsize=(10,5))
plt.barh(methods, accuracies, color=['gray', 'blue', 'green', 'red'])
plt.xlabel("Accuracy Score")
plt.title("Comparación de Feature Selection en SVM")
plt.xlim(0, 1)
plt.show()

# Identificar la mejor técnica y justificar
best_method = max(accuracy_results, key=accuracy_results.get)
print(f"La mejor técnica de selección fue: {best_method} con una precisión de {accuracy_results[best_method]:.4f}")

# Ajuste de hiperparámetros con GridSearchCV en la mejor técnica
if best_method == "Variance Threshold":
    X_train_best, X_test_best = X_train_var, X_test_var
elif best_method == "SelectKBest (chi-cuadrado)":
    X_train_best, X_test_best = X_train_kbest, X_test_kbest
elif best_method == "SelectFromModel (RandomForest)":
    X_train_best, X_test_best = X_train_sfm, X_test_sfm
else:
    X_train_best, X_test_best = X_train, X_test  # En caso de que la original sea la mejor

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_best, y_train)

print("Mejores hiperparámetros:", grid_search.best_params_)
print("Precisión optimizada:", grid_search.best_score_)

# Se utiliza accuracy_score porque mide la proporción de predicciones correctas sobre el total.
# Es ideal en este caso porque la variable objetivo 'blueWins' está balanceada (~50% victorias y derrotas).
# Si los datos estuvieran desbalanceados, métricas como precision, recall o F1-score serían más adecuadas.

# VarianceThreshold fue la mejor porque elimina características con poca varianza,
# reduciendo el ruido y mejorando la precisión del modelo sin incrementar su complejidad.
