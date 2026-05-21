# Clasificador de Árbol de Decisión — Predicción de Diabetes

> Pipeline completo de clasificación binaria sobre el dataset Pima Indians Diabetes: EDA completo con manejo de outliers y selección de características, seguido de un Árbol de Decisión entrenado y optimizado con GridSearchCV — elevando la precisión del 68,2% al 71,4%.

---

## Problema

Predecir si un paciente tiene diabetes basándose en medidas diagnósticas. El hospital quiere un modelo interpretable — uno que pueda visualizarse como un árbol de reglas clínicas de decisión, no una caja negra.

## Dataset

- **Fuente:** Dataset Pima Indians Diabetes (768 filas × 9 características)
- **Target:** `Outcome` — 1 = diabetes, 0 = no diabetes (distribución de clases 65% / 35%)
- **Características:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## Pipeline de EDA y Preprocesamiento

| Paso | Acción |
|---|---|
| Ceros imposibles | Insulin (48,7% ceros) y SkinThickness (29,6% ceros) eliminadas completamente |
| Imputación de ceros | Ceros imposibles restantes en Glucose, BloodPressure, BMI reemplazados con mediana estratificada por grupo |
| Capping de outliers | Método IQR en 3 características con sesgo positivo: Pregnancies, DiabetesPedigreeFunction, Age |
| Escalado | StandardScaler en las 6 características restantes |
| Selección de características | SelectKBest (f_classif) → top 4: **Glucose, BMI, Age, Pregnancies** |
| División | 80/20 estratificada train/test (614 entrenamiento / 154 prueba) |

**Hallazgo clave del EDA:** Glucose tiene la correlación individual más fuerte con Outcome (≈ 0,47). Los casos diabéticos se agrupan hacia la esquina de alto Glucose y alto BMI en los scatter plots — una separación clara que el árbol aprovecha.

## Resultados del Modelo

| Modelo | Precisión | Notas |
|---|---|---|
| Árbol base (parámetros por defecto) | **68,2%** | Sin podar — sobreajusta los datos de entrenamiento |
| Árbol optimizado (GridSearchCV) | **71,4%** | criterion=entropy, max_depth=5, min_samples_leaf=4 |

**GridSearchCV:** buscó criterion × max_depth × min_samples_split × min_samples_leaf con validación cruzada de 10 pliegues.

**Informe de clasificación del modelo optimizado:**

| Clase | Precisión | Recall | F1 |
|---|---|---|---|
| Sin Diabetes | 0,78 | 0,78 | 0,78 |
| Diabetes | 0,59 | 0,59 | 0,59 |

## Conclusiones Clave

- **La limpieza de datos es el trabajo real:** Insulin y SkinThickness tenían tantos ceros biológicamente imposibles (~49% y ~30%) que la imputación habría generado ruido — eliminarlas fue la decisión correcta.
- **La poda previene el sobreajuste:** El árbol sin podar memoriza las muestras de entrenamiento. Restringir `max_depth=5` obliga al modelo a aprender patrones generales y mejora la precisión en prueba en 3 puntos.
- **El desbalance de clases importa:** El modelo rinde mejor en la clase mayoritaria (Sin Diabetes). Con solo 54 casos positivos en el conjunto de prueba, el recall en la clase diabética sigue siendo el problema más difícil.

## Stack Tecnológico

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_DecisionTree_MTK.git
cd ML_DecisionTree_MTK
pip install -r requirements.txt
jupyter notebook src/DecisionTreeProject_revised.ipynb
```

El modelo entrenado se guarda en `models/` mediante `pickle`.

## Próximos Pasos

- Probar **Random Forest** o **Gradient Boosting** (XGBoost) en el mismo dataset para cuantificar la mejora de precisión del ensamblado frente a un árbol individual
- Abordar el desbalance de clases con sobremuestreo **SMOTE** o `class_weight="balanced"` y usar F1 como métrica principal en lugar de la precisión
- Añadir **valores SHAP** para explicar predicciones individuales — importante en cualquier caso de uso médico donde el razonamiento detrás de una decisión importa tanto como la propia decisión

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
