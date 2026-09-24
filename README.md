# Predicción de Quiebra Empresarial (Company Bankruptcy Prediction)

Proyecto de Machine Learning para predecir la quiebra de empresas a partir de indicadores financieros, desarrollado como Proyecto 2 del curso de Machine Learning — Facultad de Ingenierías, Ingeniería Financiera, Universidad de Medellín.

**Autores:** Joseph Garcia Llano y Jaime Castro

---

## Descripción del problema

El dataset pertenece al dominio de las **finanzas corporativas, la gestión de riesgo crediticio y la evaluación de salud financiera empresarial**. Los datos provienen del *Taiwan Economic Journal* (TEJ) y cubren el periodo 1999–2009; la condición de "quiebra" se definió con base en las regulaciones de la Bolsa de Valores de Taiwán.

- **Observaciones:** 6,819 empresas
- **Variables predictoras:** 95 indicadores financieros (rentabilidad, apalancamiento/solvencia, liquidez, eficiencia/rotación de activos, flujo de caja, crecimiento, valor por acción, banderas estructurales)
- **Variable de salida (target):** `Bankrupt?` — binaria (1 = la empresa entró en quiebra, 0 = no lo hizo)
- **Tipo de problema:** aprendizaje supervisado de **clasificación binaria**, con **fuerte desbalance de clases** (solo ~3.23% de las empresas están en quiebra)
- **Utilidad:** evaluación automatizada de riesgo de crédito (préstamos, tasas, cupos) y alerta temprana de dificultades financieras para inversionistas y reguladores

Fuente original del dataset: [Company Bankruptcy Prediction — Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)

## Metodología

1. **Análisis exploratorio de datos (EDA):** revisión de calidad de datos, clasificación de las 95 variables en categorías financieras, descripción detallada de las variables más relevantes, histogramas y boxplots interpretados.
2. **Matriz de correlación:** identificación de pares de variables con correlación > 0.8 (se conservan todas las variables, ya que los modelos usados son robustos a la multicolinealidad).
3. **Validación cruzada estratificada de 10 folds** (`StratifiedKFold`), aplicada de forma idéntica a los cinco modelos para una comparación justa.
4. **Modelos entrenados:**
   - 5.1 Árbol de decisión
   - 5.2 Comité de máquinas / Bagging (con regresión logística y con árboles)
   - 5.3 Random Forest
   - 5.4 AdaBoost
   - 5.5 XGBoost
5. **Búsqueda de hiperparámetros (Grid Search)** con `GridSearchCV`, optimizando por **F1-score** (no Accuracy, dado el desbalance de clases).
6. **Comparación final** de los modelos base vs. optimizados, selección del mejor modelo y análisis de su matriz de confusión.

## Resultados

Comparación de los 5 modelos ya optimizados (validación cruzada de 10 folds):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.9365 | 0.2710 | 0.5545 | 0.3625 | 0.7540 |
| Bagging (Árboles) | 0.9693 | 0.6291 | 0.1909 | 0.2811 | 0.9140 |
| Random Forest | 0.9588 | 0.4001 | 0.5000 | 0.4381 | 0.9408 |
| AdaBoost | 0.9720 | 0.6611 | 0.3091 | 0.4134 | 0.9307 |
| **XGBoost** | 0.9619 | 0.4371 | 0.5500 | **0.4819** | 0.9361 |

**Mejor modelo: XGBoost optimizado** (`learning_rate=0.05`, `max_depth=5`, `n_estimators=200`), con F1 = 0.4819.

De las 220 empresas que realmente quebraron, el modelo detectó correctamente 121 (55%). De las 6,599 empresas sanas, 6,438 (97.6%) fueron correctamente identificadas.

### Principales hallazgos

- Con hiperparámetros por defecto existe un trade-off entre modelos "sensibles" (árbol único, bagging con regresión logística: alto recall, baja precisión) y "conservadores" (bagging con árboles, random forest: alta precisión, bajo recall). El boosting (AdaBoost, XGBoost) evita este extremo desde el inicio.
- El ajuste de hiperparámetros tuvo el mayor impacto en Random Forest, que pasó de F1=0.22 a F1=0.44 simplemente limitando la profundidad de los árboles.
- XGBoost optimizado es el mejor modelo del proyecto en F1, aunque Random Forest optimizado queda muy cerca y logra el mejor ROC-AUC — una alternativa válida si lo que importa es el *ranking* de riesgo más que la clasificación binaria.

## Conclusiones

- Existe un trade-off consistente entre modelos sensibles y conservadores frente al desbalance de clases; el boosting secuencial (AdaBoost, XGBoost) maneja este problema mejor desde su formulación base.
- El tuning de hiperparámetros no beneficia por igual a todos los modelos: fue decisivo para Random Forest y casi irrelevante para Bagging con árboles.
- XGBoost optimizado se selecciona como modelo final por su mejor F1-score, ofreciendo el mejor equilibrio entre detectar empresas en riesgo de quiebra y controlar las falsas alarmas.

## Licencia

Este proyecto se desarrolla con fines académicos para el curso de Machine Learning de la Universidad de Medellín..
