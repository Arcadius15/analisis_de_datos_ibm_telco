# Predicción de abandono de clientes en telecomunicaciones mediante Machine Learning

**Técnica:** clasificación supervisada  
**Conjunto de datos:** IBM Telco Customer Churn (7.043 clientes)  
**Modelos:** regresión logística, KNN, árbol de decisión, Random Forest y XGBoost  
**Entorno:** Python 3.15.2 · pandas 3.1.1 · scikit-learn 1.9.1 · XGBoost 3.4.1

---

## 1. Resumen

Este proyecto desarrolla un modelo predictivo de abandono (*churn*) para una empresa de telecomunicaciones. El objetivo es **identificar a los clientes con mayor riesgo de cancelar sus servicios** para priorizar acciones de retención.

La evaluación se separó en tres conjuntos: **60 % entrenamiento, 20 % validación y 20 % prueba**. El modelo y el umbral se eligieron con validación; prueba se usó una sola vez para confirmar el resultado.

**XGBoost** fue seleccionado en validación (average precision **0,648**) y confirmó el primer lugar en prueba (**0,657**). Al contactar al 20 % de la cartera con mayor riesgo, cubre el **50,8 % de los abandonos** (lift **2,54×**). **Random Forest** quedó muy cerca (AP de prueba 0,651) y obtuvo el mejor F1 con umbral ajustado (0,629). La **regresión logística** fue competitiva en validación (0,643) y ofrece la mayor interpretabilidad.

La accuracy resultó poco informativa: KNN y la regresión logística tienen valores casi iguales (0,768 y 0,769), pero sus AP difieren de forma apreciable (0,610 y 0,627). El árbol de decisión alcanzó el mayor recall (0,797), a costa de una precision baja (0,469).

El código y los resultados completos están en `telco_churn.ipynb`. Este documento explica el problema, las decisiones metodológicas, los resultados y las conclusiones.

---

## 2. Problema y objetivos

### 2.1 Problema

Las empresas de telecomunicaciones enfrentan la pérdida de clientes, lo que afecta sus ingresos y genera mayores costos de adquisición. Se busca identificar anticipadamente a los clientes que presentan mayor probabilidad de cancelar sus servicios.

### 2.2 Objetivo general

Desarrollar un modelo predictivo de Machine Learning que permita clasificar a los clientes según su riesgo de abandono, considerando variables relacionadas con su antigüedad, facturación, tipo de contrato y servicios contratados.

### 2.3 Objetivos específicos

1. Describir y depurar el conjunto de datos, identificando valores faltantes y variables no predictivas.
2. Explorar la relación entre las características del cliente y el abandono.
3. Entrenar y ajustar cinco algoritmos de clasificación bajo un mismo esquema de validación.
4. Comparar los modelos con métricas adecuadas para clases desbalanceadas.
5. Seleccionar el modelo más adecuado para priorizar acciones de retención.

### 2.4 Alcance

- La unidad de análisis es **un cliente** y la variable objetivo es `Churn` (abandona / no abandona).
- El conjunto es **transversal**: no contiene fechas ni una ventana temporal explícita. El modelo estima el riesgo de un cliente con características similares a las observadas; **no demuestra una predicción “a 30 días”**.
- Se identifican **asociaciones predictivas**, no relaciones causales.
- Las salidas de `predict_proba` se usan como **puntuaciones de riesgo**. Al usar pesos de clase no puede suponerse que estén calibradas como probabilidades frecuentistas.

---

## 3. Conjunto de datos

El archivo `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` contiene **7.043 registros** y **21 columnas**. No hay clientes duplicados.

| Grupo | Variables | Uso |
|---|---|---|
| Identificación | `customerID` | Se excluye: identifica, no describe comportamiento |
| Demográficas | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Predictoras |
| Servicios | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Predictoras |
| Contrato y facturación | `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | Predictoras |
| Antigüedad | `tenure` (meses) | Predictora |
| Objetivo | `Churn` | `Yes` = 1, `No` = 0 |

**Distribución del objetivo**

| Clase | Clientes | Porcentaje |
|---|---:|---:|
| No abandona | 5.174 | 73,46 % |
| Abandona | 1.869 | 26,54 % |

El desbalance es moderado. Un modelo que siempre responda “no abandona” acertaría el 73,5 % de los casos, pero tendría recall 0 para la clase de interés. Por eso la **accuracy no es la métrica principal**.

**Calidad inicial**

- `TotalCharges` llega como texto porque **11 registros** contienen un espacio en blanco.
- Esos 11 clientes tienen `tenure = 0`: son clientes nuevos que aún no acumulan facturación.
- El resto de las variables no presenta valores faltantes.

**Fuente:** IBM. (2015). *Telco Customer Churn* [Conjunto de datos]. Kaggle. <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>

---

## 4. Metodología

El trabajo sigue las etapas de **CRISP-DM**: comprensión del negocio, comprensión de los datos, preparación, modelado, evaluación y despliegue (aquí, recomendaciones).

### 4.1 Los cinco algoritmos

| Algoritmo | Idea central | Papel en la comparación |
|---|---|---|
| **Regresión logística** | Modela la probabilidad de abandono con una función logística; sus coeficientes son interpretables | Línea base estadística |
| **KNN** | Clasifica según la clase mayoritaria de los clientes más parecidos | Método basado en instancias, sin supuesto global |
| **Árbol de decisión** | Divide los datos con reglas sucesivas sobre las variables | Modelo interpretable y no lineal |
| **Random Forest** | Combina muchos árboles entrenados con muestras y variables aleatorias | *Bagging*: reduce la varianza del árbol individual |
| **XGBoost** | Construye árboles de forma secuencial, corrigiendo los errores del anterior | *Boosting*: alto desempeño en datos tabulares |

### 4.2 Esquema de evaluación

1. **Partición estratificada 60/20/20**:
   - **Entrenamiento (60 %):** ajuste de hiperparámetros con validación cruzada de 5 particiones.
   - **Validación (20 %):** selección del modelo y ajuste del umbral.
   - **Prueba (20 %):** evaluación final, una sola vez.
2. **Métrica principal: average precision (AP)**. Es un resumen no interpolado de la curva precisión–recall: pondera la precision en cada umbral por el incremento de recall. No es el área trapezoidal de la curva.
3. **Métricas complementarias:** recall, precision, F1, ROC-AUC y accuracy.
4. **Ajuste del umbral** maximizando F1 sobre validación.
5. **Evaluación final** sobre prueba, sin cambiar el modelo ni el umbral después de verla.

### 4.3 Prevención de fuga de información

Toda imputación, codificación y estandarización se aprende **dentro de cada partición de entrenamiento** mediante `Pipeline` y `ColumnTransformer`.

La partición se realiza **antes del análisis exploratorio y de cualquier transformación**. El conjunto de prueba no participa en la selección de variables, hiperparámetros, modelo ni umbral.

---

## 5. Entorno y ejecución

El análisis se ejecuta con **Python 3.15.2** y las dependencias están fijadas en `pyproject.toml` y `uv.lock`.

```bash
# 1) Crear o sincronizar el entorno
uv sync

# 2) Registrar el kernel de Python 3.15 para Jupyter
uv run python -m ipykernel install --user \
    --name telco-churn \
    --display-name "Python 3.15 (Telco Churn)"

# 3) Abrir JupyterLab
uvx --python 3.14 --with jupyterlab jupyter lab
```

**Dependencias principales:** `pandas`, `matplotlib`, `seaborn`, `scikit-learn` y `xgboost`. Como dependencias de desarrollo: `ipykernel` y `nbconvert`.

> **Nota técnica:** el kernel que ejecuta el notebook es Python 3.15.2. El servidor de JupyterLab se inicia con Python 3.14 porque `PyYAML`, dependencia del servidor, todavía no publica una rueda binaria para 3.15 y requeriría compilar con Visual C++. Todo el código de análisis, los modelos y las métricas se ejecutan íntegramente en 3.15.

---

## 6. Desarrollo del análisis

El notebook contiene 22 celdas de código, organizadas en 17 bloques temáticos. Esta sección explica las decisiones que no resultan evidentes al leer el código.

### 6.1 Depuración

1. **`TotalCharges`**: se convierte a numérico. Los 11 faltantes se imputan con **0** dentro del pipeline, no con la mediana, porque corresponden a clientes nuevos sin facturación acumulada.
2. **Validación de la regla:** un `assert` comprueba que **todos** los faltantes tengan `tenure = 0`. Si apareciera un faltante con antigüedad mayor, el proceso se detiene para revisarlo.
3. **`customerID`**: se excluye de los predictores. Es un identificador único sin poder predictivo.
4. **`Churn`**: se convierte a 0/1.
5. **Sin eliminación de filas**: se conservan los 7.043 clientes.
6. **Categorías `No internet service`**: son información válida, no valores faltantes; se conservan como categoría.

### 6.2 Partición

| Subconjunto | Clientes | Tasa de abandono | Uso |
|---|---:|---:|---|
| Entrenamiento | 4.225 | 26,53 % | Ajuste de hiperparámetros |
| Validación | 1.409 | 26,54 % | Selección de modelo y umbral |
| Prueba | 1.409 | 26,54 % | Evaluación final |

La estratificación conserva la proporción de clases en los tres subconjuntos.

### 6.3 Análisis exploratorio (solo entrenamiento)

Se comparan **tasas de abandono**, no solo conteos: un segmento grande puede acumular muchos abandonos sin tener mayor riesgo relativo.

| Variable | Categoría | Tasa de abandono |
|---|---|---:|
| Contrato | Mes a mes | **43,1 %** |
| | Un año | 10,8 % |
| | Dos años | 2,5 % |
| Internet | Fibra óptica | **42,3 %** |
| | DSL | 18,7 % |
| | Sin internet | 7,3 % |
| Método de pago | Cheque electrónico | **45,6 %** |
| | Cheque enviado | 19,5 % |
| | Transferencia automática | 16,2 % |
| | Tarjeta automática | 15,1 % |
| Soporte técnico | No | **41,8 %** |
| | Sí | 15,0 % |
| | Sin internet | 7,3 % |

Además:

- Los abandonos se concentran en los **primeros meses** de antigüedad.
- Los clientes que abandonan tienden a pagar **cargos mensuales más altos**, aunque hay solapamiento entre grupos.
- La combinación **contrato mes a mes + fibra óptica** presenta las tasas más altas.

Estas son asociaciones descriptivas; el modelo determina después cuánto aporta cada variable en conjunto.

### 6.4 Preparación de variables

| Rama | Variables | Transformación |
|---|---|---|
| Numéricas | `tenure`, `MonthlyCharges` | Imputación por mediana + estandarización |
| Facturación acumulada | `TotalCharges` | Imputación con 0 + estandarización |
| Categóricas | 16 variables | Imputación por moda + *one-hot encoding* |

La estandarización es imprescindible para KNN y regresión logística, y no perjudica a los árboles. Se aplica la misma preparación a los cinco modelos para que la comparación sea justa. `OneHotEncoder(handle_unknown="ignore")` evita errores con categorías no vistas.

### 6.5 Desbalance

- **Regresión logística, árbol y Random Forest:** `class_weight="balanced"`.
- **XGBoost:** `scale_pos_weight` con la proporción de negativos y positivos del entrenamiento.
- **KNN:** no admite ponderación por clase; se compensa con el ajuste del umbral.

Además, para todos los modelos se ajusta el umbral de decisión en validación. El desbalance no se resuelve solo con pesos: también se decide **a partir de qué puntuación se alerta a un cliente**.

---

## 7. Resultados

### 7.1 Línea base

| Métrica | Resultado |
|---|---:|
| Average precision | 0,265 |
| ROC-AUC | 0,500 |
| Recall | 0,000 |
| Precision | 0,000 |
| F1 | 0,000 |
| Accuracy | **0,735** |

La línea base “siempre no abandona” alcanza 73,5 % de accuracy con recall 0. Es la evidencia directa de que la accuracy sola no sirve en este problema.

### 7.2 Hiperparámetros seleccionados

| Modelo | Mejores hiperparámetros |
|---|---|
| Regresión logística | `C = 10,0` |
| KNN | `n_neighbors = 31`, `weights = uniform` |
| Árbol de decisión | `max_depth = 7`, `min_samples_leaf = 20` |
| Random Forest | `max_depth = 8`, `min_samples_leaf = 5` |
| XGBoost | `max_depth = 3`, `learning_rate = 0,05` |

En la validación cruzada del entrenamiento, XGBoost y Random Forest quedaron prácticamente empatados (AP 0,670 y 0,670), seguidos por regresión logística (0,665), KNN (0,623) y árbol de decisión (0,617).

### 7.3 Validación y selección del modelo

| Modelo | AP | ROC-AUC | Recall @0,5 | Precision @0,5 | F1 @0,5 | Accuracy @0,5 |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **0,648** | 0,838 | 0,765 | 0,517 | 0,617 | 0,748 |
| Regresión logística | 0,643 | 0,837 | 0,786 | 0,512 | 0,620 | 0,744 |
| Random Forest | 0,638 | 0,838 | 0,786 | 0,510 | 0,619 | 0,743 |
| Árbol de decisión | 0,601 | 0,818 | 0,773 | 0,495 | 0,603 | 0,730 |
| KNN | 0,595 | 0,824 | 0,561 | 0,621 | 0,590 | 0,793 |

**XGBoost fue seleccionado por tener la mayor AP en validación.** La regresión logística quedó a solo 0,005 y Random Forest a 0,010. El conjunto de prueba no participó en esta decisión.

### 7.4 Umbral ajustado en validación

Se eligió, para cada modelo, el umbral que maximiza F1 en validación:

| Modelo | Umbral | F1 (validación) | Recall (validación) | Precision (validación) |
|---|---:|---:|---:|---:|
| Random Forest | 0,578 | 0,638 | 0,727 | 0,568 |
| Regresión logística | 0,616 | 0,637 | 0,701 | 0,584 |
| XGBoost | 0,603 | 0,633 | 0,693 | 0,583 |
| KNN | 0,419 | 0,624 | 0,701 | 0,562 |
| Árbol de decisión | 0,472 | 0,611 | 0,824 | 0,485 |

Maximizar F1 es un criterio académico. En una aplicación real, el umbral debería fijarse según la capacidad de contacto y los costos de retención.

### 7.5 Evaluación final en prueba

| Modelo | AP | ROC-AUC | Recall* | Precision* | F1* | Accuracy* |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **0,657** | 0,841 | 0,701 | 0,563 | 0,625 | 0,776 |
| Regresión logística | 0,627 | **0,842** | 0,703 | 0,551 | 0,618 | 0,769 |
| Random Forest | 0,651 | **0,842** | 0,719 | 0,559 | **0,629** | 0,775 |
| Árbol de decisión | 0,600 | 0,822 | **0,797** | 0,469 | 0,591 | 0,707 |
| KNN | 0,610 | 0,830 | 0,693 | 0,550 | 0,613 | 0,768 |

\* Umbral ajustado en validación.

**Lectura:**

- **XGBoost confirma el primer lugar en prueba** (AP 0,657), seguido por Random Forest (0,651) y regresión logística (0,627).
- **Random Forest** logra el mejor F1 (0,629) y un recall alto (0,719), con precision 0,559.
- **El árbol de decisión** alcanza el mayor recall (0,797), pero con precision 0,469: alerta a muchos clientes que no abandonan.
- **KNN** tiene accuracy 0,768, similar a la regresión logística (0,769), pero su AP es menor (0,610 frente a 0,627). La accuracy no distingue bien entre modelos.

### 7.6 Capacidad operativa: mismos 282 clientes para todos

Se contacta exactamente al **20 % de la cartera de prueba**: **282 de 1.409 clientes**, para todos los modelos. La selección se hace ordenando por puntuación; el desempate es estable.

| Modelo | Abandonos detectados | Cobertura | Precision en el grupo | Lift vs. azar |
|---|---:|---:|---:|---:|
| **XGBoost** | **190** | **50,8 %** | **67,4 %** | **2,54×** |
| Regresión logística | 188 | 50,3 % | 66,7 % | 2,51× |
| Random Forest | 186 | 49,7 % | 66,0 % | 2,49× |
| KNN | 183 | 48,9 % | 64,9 % | 2,45× |
| Árbol de decisión | 176 | 47,1 % | 62,4 % | 2,35× |

XGBoost detecta **190 de los 374 abandonos** del conjunto de prueba con 282 contactos. Es **2,54 veces más** que una selección aleatoria del mismo tamaño.

La diferencia entre XGBoost, regresión logística y Random Forest es pequeña (190, 188 y 186 abandonos). Con una sola partición no puede afirmarse que sea una ventaja estable.

### 7.7 Curvas y matrices de confusión

- Las curvas **ROC** de XGBoost, Random Forest y regresión logística son prácticamente indistinguibles (AUC 0,841–0,842); árbol y KNN quedan por debajo (0,822 y 0,830).
- Las curvas **precisión–recall** separan mejor a XGBoost (AP 0,657) de KNN (0,610) y del árbol (0,600).
- En las matrices con umbral ajustado, el árbol detecta cerca del 80 % de los abandonos, pero con menos de la mitad de sus alertas correctas. XGBoost detecta alrededor del 70 %, con una precision cercana al 56 %.

El umbral ajustado de XGBoost en validación (0,603) produjo en prueba un recall de 0,701 y una precision de 0,563. No se modificó después de ver estos resultados.

### 7.8 Interpretación de variables

**Random Forest: variables más importantes**

| Variable | Importancia |
|---|---:|
| `Contract_Month-to-month` | 0,152 |
| `tenure` | 0,123 |
| `TotalCharges` | 0,083 |
| `Contract_Two year` | 0,076 |
| `OnlineSecurity_No` | 0,073 |
| `TechSupport_No` | 0,053 |
| `MonthlyCharges` | 0,053 |
| `InternetService_Fiber optic` | 0,049 |

**Regresión logística: coeficientes condicionales**

| Mayor riesgo | Coeficiente | Menor riesgo | Coeficiente |
|---|---:|---|---:|
| `InternetService_Fiber optic` | +1,162 | `MonthlyCharges` | −1,581 |
| `Contract_Month-to-month` | +0,722 | `tenure` | −1,271 |
| `TotalCharges` | +0,612 | `InternetService_DSL` | −0,944 |
| `StreamingTV_Yes` | +0,499 | `Contract_Two year` | −0,899 |
| `StreamingMovies_Yes` | +0,472 | `InternetService_No` | −0,460 |

**Precaución interpretativa:** los coeficientes son **condicionales**. `tenure`, `MonthlyCharges` y `TotalCharges` están fuertemente correlacionadas y `TotalCharges ≈ tenure × MonthlyCharges`; al controlar por las otras variables, el signo de `MonthlyCharges` puede invertirse (efecto de supresión). Por eso:

- Los coeficientes describen el comportamiento del modelo, **no efectos causales aislados**.
- La importancia de Random Forest se basa en impureza y favorece variables continuas o de alta cardinalidad; sirve como orientación, no como prueba de causalidad.
- Para explicar el negocio conviene complementar con análisis descriptivos y, idealmente, experimentos.

---

## 8. Conclusiones

### 8.1 ¿Cuál es el modelo más adecuado?

**XGBoost es la recomendación principal** para priorizar clientes según su riesgo de abandono:

- Fue seleccionado por la mayor **average precision en validación** (0,648).
- Confirmó el primer lugar en **prueba** (AP 0,657).
- Obtuvo la mayor **cobertura de abandonos en el top 20 %** (50,8 %, lift 2,54×).
- Presentó un equilibrio razonable entre recall (0,701) y precision (0,563) con el umbral ajustado.

La evidencia no muestra una superioridad contundente:

- **Random Forest** es la alternativa más cercana: AP de prueba 0,651, el mejor F1 (0,629) y recall 0,719. Si se prioriza cobertura con un F1 competitivo, es una elección defendible.
- **Regresión logística** quedó a solo 0,005 de AP en validación y ofrece coeficientes interpretables, bajo costo de mantenimiento y facilidad para explicar cada alerta. En prueba quedó más atrás (0,627), pero sigue siendo una opción razonable si la interpretabilidad es requisito.
- **Árbol de decisión** logra el mayor recall (0,797), pero su precision (0,469) y su AP (0,600) son bajas; generaría muchas alertas innecesarias.
- **KNN** tiene una AP de 0,610 y no supera a los tres primeros en ningún criterio relevante para retención.

Con una sola partición de validación y otra de prueba, las diferencias entre los tres primeros son pequeñas: 0,005–0,010 de AP en validación y 0,006–0,030 en prueba. No se realizó una prueba de significancia estadística. Por eso **XGBoost es el mejor candidato con la evidencia disponible, pero no puede afirmarse una superioridad concluyente**. Una validación repetida o temporal permitiría decidir con mayor solidez.

### 8.2 Limitaciones

1. **Datos transversales:** no permiten afirmar que el modelo predice el abandono “en los próximos 30 días”; estiman riesgo con base en el perfil actual.
2. **Asociación, no causalidad:** los coeficientes y las importancias describen el modelo, no el efecto de intervenir una variable.
3. **Desbalance moderado:** el 26,5 % de abandonos obliga a usar AP, recall y umbrales ajustados; la accuracy sola sería engañosa.
4. **Umbral dependiente del negocio:** el umbral óptimo cambia si cambia el costo de contactar o el valor de retener.
5. **Calibración:** las puntuaciones no están calibradas; no deben interpretarse como probabilidades exactas de abandono.
6. **Sin validación temporal:** un despliegue real debería entrenarse con datos históricos y evaluarse con clientes posteriores.
7. **Una sola partición de validación y prueba:** los resultados tienen variabilidad; conviene validación repetida o temporal antes de decidir.
8. **Variables limitadas:** el conjunto no incluye interacciones con el servicio, reclamos, competencia ni contexto económico.

### 8.3 Recomendaciones

- Usar el modelo como **priorizador de campañas de retención**, no como decisión automática de cancelación.
- Revisar el umbral con el área de retención según su capacidad operativa: con 20 % de la cartera se cubre aproximadamente la mitad de los abandonos.
- Reentrenar periódicamente y monitorear la deriva de los datos.
- Complementar con experimentos o análisis causales antes de concluir que una acción evita el abandono.
- Si se requiere comunicación de probabilidades, evaluar una etapa de calibración y validarla con datos posteriores.

---

## 9. Estructura del notebook

`telco_churn.ipynb` contiene únicamente código comentado, sin celdas de texto. Los bloques son:

| Bloque | Contenido |
|---:|---|
| 1 | Configuración e importaciones |
| 2 | Carga y revisión inicial |
| 3 | Depuración |
| 4 | Partición 60/20/20 |
| 5 | Análisis exploratorio |
| 6 | Preparación de variables |
| 7 | Modelos y grillas |
| 8 | Línea base |
| 9 | Ajuste y validación |
| 10 | Ajuste del umbral |
| 11 | Evaluación en prueba |
| 12 | Capacidad operativa (282 clientes) |
| 13 | Curvas ROC y precisión–recall |
| 14 | Matrices de confusión |
| 15 | Coeficientes de la regresión logística |
| 16 | Importancia de variables en Random Forest |
| 17 | Resumen: selección y confirmación |

---

## 10. Referencias

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21–27.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). Wiley.
- IBM. (2015). *Telco Customer Churn* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
