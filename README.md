# Laboratorio3-IA
Michelle Mejía 22596 y  Silvia Illescas 22376

# Proyecto: Análisis de Modelos y Selección de Características
Este proyecto está enfocado en el análisis de modelos de clasificación y la selección de características utilizando el dataset de League of Legends. A lo largo de este proyecto, se implementan técnicas de feature selection para mejorar la precisión y rendimiento de modelos de machine learning.

## Tareas Realizadas:
1. Clasificación Binaria con Perceptrón
Descripción: Utilizamos el dataset de Iris para realizar una clasificación binaria, seleccionando dos características: sepal length y sepal width. Implementamos el perceptrón de una sola capa para clasificar las muestras y visualizamos la frontera de decisión aprendida por el modelo.
Métrica de Desempeño: Utilizamos accuracy como métrica de desempeño debido a la simplicidad del problema y la naturaleza balanceada del dataset. El modelo alcanzó un accuracy de 1.00, lo que indica que clasificó correctamente todas las muestras.

2. Selección de Características (Feature Selection)
Descripción: En esta tarea, aplicamos al menos tres técnicas de selección de características distintas para el análisis de un dataset de League of Legends. Después de realizar la selección, ajustamos el modelo Support Vector Machine (SVM) para clasificar las partidas de League of Legends.

Las tres técnicas de feature selection utilizadas son:

Selección basada en la importancia de características: Usamos un modelo como un árbol de decisión para obtener la importancia de cada característica y seleccionamos las más relevantes.

Selección recursiva de características (RFE): Seleccionamos características eliminando de manera recursiva aquellas que menos impactan el rendimiento del modelo.

Selección por correlación: Eliminamos características altamente correlacionadas, manteniendo solo las más independientes.

Métrica de Desempeño: Para evaluar el rendimiento de los modelos después de la selección de características, usamos la precisión (accuracy). Esta métrica fue elegida porque mide directamente la capacidad del modelo para predecir correctamente el resultado de las partidas (objetivo: blueWins).

Análisis de Resultados: Comparamos los resultados obtenidos con las tres técnicas de selección de características y evaluamos cuál versión hizo un mejor ajuste al modelo, considerando el rendimiento en términos de precisión y otros factores de rendimiento, como el tiempo de ejecución.

3. Ajuste de Parámetros (Parameter Tuning)
Descripción: Después de realizar la selección de características, ajustamos los parámetros del modelo Support Vector Machine (SVM) usando técnicas como la búsqueda en cuadrícula (GridSearchCV). Este ajuste optimiza el rendimiento del modelo, asegurando que se utilicen las mejores combinaciones de parámetros para obtener un modelo más preciso.

Conclusiones:
La selección de características mejora la precisión y la eficiencia del modelo al reducir la cantidad de datos y enfocarse solo en las variables más relevantes.
El uso de Support Vector Machine (SVM) en combinación con la selección de características y el ajuste de parámetros resultó en una mejora significativa en la precisión y en el rendimiento general del modelo.
