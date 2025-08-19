# Predicción de Abandono (Churn) y Segmentación de Clientes (Gimnasio Model Fitness)

## Descripción del Proyecto
Este proyecto se enfoca en el análisis del abandono de clientes para la cadena de gimnasios Model Fitness, con el objetivo de desarrollar una estrategia de retención basada en datos. Incluye la predicción de la probabilidad de abandono para cada cliente, la creación de perfiles de usuarios típicos y el análisis de los factores más influyentes en el abandono.

## Problema de Negocio / Objetivo
Model Fitness busca combatir la pérdida de clientes y mejorar sus estrategias de interacción. Para ello, fue necesario:
1.  **Predecir la probabilidad de abandono** (churn) para cada cliente en el próximo mes.
2.  **Identificar y caracterizar grupos de usuarios** (clústeres) con comportamientos y propensiones de abandono distintivos.
3.  **Analizar los factores** que tienen mayor impacto en el abandono de clientes.
4.  **Elaborar conclusiones y recomendaciones** prácticas para mejorar la atención al cliente y reducir la rotación.

## Conjunto de Datos
El análisis se basó en un dataset que contiene datos de clientes de un mes en particular, así como información del mes anterior:
* **`gym_churn_us.csv`**:
    * `Churn`: Variable objetivo (0 = no abandona, 1 = abandona).
    * Datos demográficos y de membresía (`gender`, `Near_Location`, `Partner`, `Promo_friends`, `Phone`, `Age`, `Lifetime`, `Contract_period`, `Month_to_end_contract`).
    * Datos de uso (`Group_visits`, `Avg_class_frequency_total`, `Avg_class_frequency_current_month`, `Avg_additional_charges_total`).

## Herramientas y Tecnologías Utilizadas
* **Python:** Lenguaje principal para todo el proceso de análisis, modelado y visualización.
    * `pandas`: Para la carga, limpieza, manipulación y análisis exploratorio de datos.
    * `numpy`: Para operaciones numéricas.
    * `matplotlib` / `seaborn`: Para la visualización de distribuciones, histogramas y matrices de correlación.
    * `sklearn` (scikit-learn): Para el preprocesamiento de datos (escalado), construcción de modelos de clasificación (`LogisticRegression`, `RandomForestClassifier`), evaluación de modelos (Accuracy, Precision, Recall) y clustering (`KMeans`).
    * `scipy.cluster.hierarchy`: Para la creación de dendrogramas.
* **Jupyter Notebook - VSCode:** Entornos interactivo para el desarrollo del análisis.

## Metodología y Análisis
### Parte 1: Análisis Exploratorio de Datos (EDA)
1.  **Inspección del Dataset:** Verificación de valores ausentes, tipos de datos y estadísticas descriptivas (`.describe()`).
2.  **Análisis Comparativo por Grupo de Abandono:** Estudio de los valores medios de las características para clientes que abandonaron vs. los que se quedaron (`.groupby()`).
3.  **Visualizaciones:** Creación de histogramas y distribuciones de características para ambos grupos (abandonan/no abandonan) para identificar diferencias clave.
4.  **Matriz de Correlación:** Visualización de las correlaciones entre todas las características para entender las relaciones entre variables.

### Parte 2: Construcción de un Modelo Predictivo de Abandono
1.  **Preparación del Conjunto de Datos:** División de los datos en conjuntos de entrenamiento y validación (`train_test_split`).
2.  **Entrenamiento del Modelo:** Se entrenaron dos modelos de clasificación binaria:
    * **Regresión Logística.**
    * **Bosque Aleatorio (Random Forest Classifier).**
3.  **Evaluación del Modelo:** Cálculo y comparación de métricas clave (exactitud, precisión y *recall*) en el conjunto de validación para ambos modelos, identificando el de mejor rendimiento para la predicción de abandono.

### Parte 3: Creación de Clústeres de Usuarios
1.  **Preprocesamiento para Clustering:** Estandarización de los datos (excluyendo la variable objetivo 'Churn').
2.  **Determinación del Número Óptimo de Clústeres:** Utilización de `linkage()` para crear una matriz de distancias y trazar un dendrograma, lo que permite visualizar la estructura de los datos y estimar un número adecuado de clústeres.
3.  **Entrenamiento de K-Means:** Aplicación del algoritmo K-Means para agrupar a los usuarios en clústeres (ej., n=5).
4.  **Análisis de Clústeres:**
    * Análisis de los valores medios de las características para cada clúster para entender sus perfiles distintivos.
    * Visualización de las distribuciones de características por clúster.
    * Cálculo de la **tasa de abandono para cada clúster** para identificar cuáles son propensos a irse y cuáles son leales.

* Las **medidas recomendadas para reducir la rotación** incluyen:
    * **Programas de incentivo** para clientes con contratos de 1 mes o próximos a vencer.
 
    * <img width="852" height="548" alt="image" src="https://github.com/user-attachments/assets/dbaf4b49-2b3c-44f9-ae51-7f17bf93e6ca" />

    <img width="1288" height="889" alt="image" src="https://github.com/user-attachments/assets/d02d4a26-3fb2-4dd2-94e9-c741cb280d30" />

    <img width="1241" height="856" alt="image" src="https://github.com/user-attachments/assets/0d761441-0b9c-45b3-928e-12752f18f49f" />

    <img width="852" height="549" alt="image" src="https://github.com/user-attachments/assets/2d6c828c-03c8-4ecb-b574-9a5f9e614aa1" />

  <img width="858" height="544" alt="image" src="https://github.com/user-attachments/assets/a994f06d-d4db-4191-b1db-1af728caadee" />

  <img width="704" height="470" alt="image" src="https://github.com/user-attachments/assets/050f545a-bdf1-425b-8224-f054f0325c90" />

  <img width="406" height="480" alt="image" src="https://github.com/user-attachments/assets/6c449639-064a-422e-aa64-b86fd33ae676" />

  







    

    * **Fomentar la participación en sesiones grupales** y aumentar la frecuencia de visitas.
    * **Ofrecer descuentos especiales** a empleados de compañías asociadas con baja participación.
* Se observó que [describir cualquier otro patrón relevante, ej. clientes que gastan más en servicios adicionales tienden a ser más leales].
