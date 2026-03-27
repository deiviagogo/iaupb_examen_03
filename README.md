# iaupb_examen_03

## Identificación del estudiante

David Vallejo

000551513

david.vallejog@upb.edu.co


| Campo | Detalle |
|---|---|
| **Institución** | Universidad Pontificia Bolivariana |
| **Programa** | Ingeniería en Sistemas e Informática |
| **Curso** | Inteligencia Artificial |
| **Docente** | Juan Darío Rodas |
| **Evaluación** | Examen No. 3  |
| **Fecha de entrega** | Viernes 27 de marzo de 2026|

---

## Contenido del repositorio

```
iaupb_examen_03/
├── README.md                                  ← este archivo
└── iaupb_examen_03_clasificacion.ipynb        ← solución completa en Jupyter Notebook
```

---

## Descripción del problema

Se comparan tres algoritmos de clasificación supervisada aplicados al dataset **Palmer Penguins** (UCI ML Repository, 344 filas, 6 características, 3 clases):

| Algoritmo | Tipo de frontera |
|---|---|
| Logistic Regression | Lineal |
| Random Forest | No lineal (ensamble de árboles) |
| SVM (kernel RBF) | No lineal |

**Variable objetivo:** especie del pingüino (`Adelie`, `Chinstrap`, `Gentoo`)

---

## Estructura del Notebook

El notebook `iaupb_examen_03_clasificacion.ipynb` sigue el flujo completo del proceso de aprendizaje supervisado:

### 1. Importación de librerías
Se importan `pandas`, `numpy`, `matplotlib`, `seaborn` y los módulos de `sklearn` necesarios. Se fija `SEED = 42` para reproducibilidad.

### 2. Carga y exploración inicial
El dataset se carga desde el paquete `palmerpenguins`. Se verifica forma, tipos de datos, estadísticos descriptivos y conteo de valores faltantes (~19 NaN distribuidos entre variables numéricas y `sex`).

### 3. EDA – Análisis Exploratorio
- **Distribución de clases**: gráfico de barras con frecuencias absolutas por especie.
- **Boxplots por variable numérica**: uno por cada variable (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`), diferenciados por especie con color.
- **Matriz de correlación**: heatmap con coeficientes de Pearson en rango [-1, 1].

### 4. Preprocesamiento
- División **80% train / 20% test** con `random_state=42` y `stratify=y`.
- **Pipeline numérico**: imputación con mediana → estandarización (`StandardScaler`).
- **Pipeline categórico**: imputación con moda → codificación (`OneHotEncoder`, `drop='first'`).
- `ColumnTransformer` aplica `fit_transform` solo sobre train y `transform` sobre test → **sin data leakage**.

### 5. Entrenamiento de modelos
Los tres modelos se entrenan con los mismos datos transformados:
- `LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')`
- `RandomForestClassifier(n_estimators=200)`
- `SVC(kernel='rbf', probability=True)`

### 6. Evaluación de métricas
Tabla comparativa con `accuracy`, `precision`, `recall` y `F1-score` (promedio ponderado) para los tres modelos. Se incluye `classification_report` completo por clase.

### 7. Matrices de confusión
Una figura independiente por algoritmo, con etiquetas de especie en los ejes y valores absolutos en cada celda.

### 8. Comparación visual de métricas
Gráfico de barras agrupadas que consolida las cuatro métricas para los tres algoritmos simultáneamente (eje Y de 0 a 1).

### 9. Selección del mejor modelo
Análisis comparativo basado en el conjunto completo de métricas, con justificación integral del modelo ganador.

### 10. Preguntas de análisis e interpretación
Siete preguntas respondidas como celdas de texto dentro del notebook, una por celda.

---

## Por qué los tres modelos obtienen métricas de 1.0

Los tres algoritmos alcanzan `accuracy = 1.0`, `precision = 1.0`, `recall = 1.0` y `F1 = 1.0`. Las matrices de confusión no presentan ningún error fuera de la diagonal. Este resultado, aunque llamativo, **es genuino** para este dataset con esta configuración de preprocesamiento.

### Razón 1: el dataset Palmer Penguins es altamente separable

A diferencia de datasets como Iris, donde Versicolor y Virginica se solapan considerablemente, las tres especies de pingüinos quedan bien diferenciadas en el espacio de características cuando se combinan todas las variables disponibles:

- **Gentoo** se separa trivialmente del resto por `flipper_length_mm` y `body_mass_g` (es notablemente más grande).
- **Adelie vs Chinstrap**: aunque se solapan en algunas variables individuales, la combinación de `bill_length_mm` + `bill_depth_mm` + variables geográficas (`island`) las separa con alta confianza.

Con 6 características y solo 3 clases razonablemente diferenciadas, los modelos no lineales (Random Forest, SVM RBF) tienen suficiente capacidad para encontrar fronteras de decisión perfectas en el conjunto de prueba de 69 registros.

### Razón 2: la variable `island` es un predictor casi determinístico

El dataset incluye `island` como variable predictora. En los datos recolectados:

- **Torgersen** → exclusivamente Adelie
- **Biscoe** → mayoritariamente Gentoo
- **Dream** → mezcla de Adelie y Chinstrap

Incluir `island` hace que los modelos puedan aprender una regla geográfica casi perfecta antes de necesitar siquiera las variables morfológicas. Esto eleva el techo de desempeño de manera artificial para este dataset específico.

### Razón 3: partición estratificada con `random_state=42`

La división 80/20 con `stratify=y` y semilla fija produce exactamente los mismos 69 ejemplos de prueba en cada ejecución. Con un dataset tan separable, esos 69 registros resultan todos clasificables correctamente por los tres modelos.

### Conclusión

El 1.0 **no es un error de implementación ni data leakage**. Es el resultado real de aplicar tres algoritmos de clasificación robustos sobre un dataset que, con el conjunto completo de variables (incluyendo `island`), es linealmente o casi linealmente separable. El pipeline de preprocesamiento está correctamente implementado: `fit_transform` solo sobre train, `transform` sobre test, sin fuga de información.

Para obtener métricas menores a 1.0 habría que excluir `island` de las variables predictoras, lo que forzaría a los modelos a depender exclusivamente de las mediciones morfológicas. En ese escenario se esperarían valores de accuracy entre 0.95 y 0.99, con algunos errores entre Adelie y Chinstrap.

---

## Notas técnicas del preprocesamiento

- **Imputación numérica con mediana**: más robusta que la media ante los valores atípicos observados en `body_mass_g` y `bill_length_mm`.
- **Imputación categórica con moda**: para `island` y `sex`, la categoría más frecuente es la estimación más conservadora.
- **Sin data leakage**: el `ColumnTransformer` llama a `fit_transform` exclusivamente sobre `X_train`. El `X_test` solo pasa por `transform`, garantizando que los estadísticos (media, desvío, categorías) se calculen únicamente sobre datos de entrenamiento.
- **Stratify en la división**: preserva las proporciones de clase (Adelie ~44%, Gentoo ~36%, Chinstrap ~20%) tanto en train como en test.

---

## Dataset

**Palmer Penguins – Archipiélago Palmer, Antártica**  
Fuente: https://archive.ics.uci.edu/dataset/690/palmer+penguins-3  
344 registros · 6 características · 3 clases (`Adelie`, `Chinstrap`, `Gentoo`)
