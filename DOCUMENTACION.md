# Predicción de abandono de clientes en telecomunicaciones mediante Machine Learning

**Técnica:** clasificación supervisada  
**Conjunto de datos:** IBM Telco Customer Churn (7.043 clientes)  
**Modelos:** regresión logística, KNN, árbol de decisión, Random Forest y XGBoost  
**Entorno:** Python 3.15.2 · pandas 3.1.1 · scikit-learn 1.9.1 · XGBoost 3.4.1

---

## 1. Resumen

Este proyecto desarrolla un modelo predictivo de abandono (*churn*) para una empresa de telecomunicaciones. El objetivo es **identificar a los clientes con mayor probabilidad de cancelar sus servicios** para priorizar acciones de retención.

Se compararon cinco algoritmos de clasificación bajo el mismo esquema de validación. El mejor resultado lo obtuvo **XGBoost**, con una *average precision* de **0,664** en el conjunto de prueba y una cobertura del **52,1 % de los abandonos** al contactar al 20 % de la cartera con mayor riesgo. **Random Forest** logró el mejor F1 con umbral ajustado (0,630) y un recall alto (0,767), mientras que la **regresión logística** quedó muy cerca en capacidad predictiva (AP 0,633) y ofrece la mayor interpretabilidad.

Un hallazgo relevante es que **KNN obtuvo la mayor accuracy (0,774) pero la menor average precision (0,611)**: elegir por accuracy habría seleccionado el modelo menos útil para detectar abandonos.

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

1. **Partición estratificada 80/20**: 80 % para entrenamiento y validación, 20 % para prueba.
2. **Validación cruzada estratificada de 5 particiones** sobre el entrenamiento.
3. **Métrica principal: average precision (AP)**, área bajo la curva precisión–recall. Es más informativa que ROC-AUC cuando hay desbalance.
4. **Métricas complementarias:** recall, precision, F1, ROC-AUC y accuracy.
5. **Ajuste del umbral** con predicciones *out-of-fold* del entrenamiento.
6. **Evaluación final única** sobre el 20 % de prueba, que no participa en ninguna decisión.

### 4.3 Prevención de fuga de información

Toda imputación, codificación y estandarización se aprende **dentro de cada partición de entrenamiento** mediante `Pipeline` y `ColumnTransformer`. Si se imputara o escalara antes de dividir, el modelo vería información del conjunto de prueba y las métricas serían artificialmente optimistas.

La partición se realiza **antes del análisis exploratorio y de cualquier transformación**.

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
2. **`customerID`**: se excluye de los predictores. Es un identificador único sin poder predictivo.
3. **`Churn`**: se convierte a 0/1.
4. **Sin eliminación de filas**: se conservan los 7.043 clientes.
5. **Categorías `No internet service`**: son información válida, no valores faltantes; se conservan como categoría.

### 6.2 Partición

| Subconjunto | Clientes | Tasa de abandono |
|---|---:|---:|
| Entrenamiento | 5.634 | 26,54 % |
| Prueba | 1.409 | 26,54 % |

La estratificación conserva la proporción de clases en ambos subconjuntos.

### 6.3 Análisis exploratorio (solo entrenamiento)

Se comparan **tasas de abandono**, no solo conteos: un segmento grande puede acumular muchos abandonos sin tener mayor riesgo relativo.

| Variable | Categoría | Tasa de abandono |
|---|---|---:|
| Contrato | Mes a mes | **42,7 %** |
| | Un año | 11,1 % |
| | Dos años | 2,9 % |
| Internet | Fibra óptica | **42,1 %** |
| | DSL | 18,7 % |
| | Sin internet | 7,2 % |
| Método de pago | Cheque electrónico | **45,7 %** |
| | Cheque enviado | 19,3 % |
| | Transferencia automática | 16,2 % |
| | Tarjeta automática | 14,9 % |
| Soporte técnico | No | **41,8 %** |
| | Sí | 15,2 % |
| | Sin internet | 7,2 % |

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

Además, para todos los modelos se ajusta el umbral de decisión en validación. El desbalance no se resuelve solo con pesos: también se decide **a partir de qué probabilidad se alerta a un cliente**.

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

| Modelo | Mejores hiperparámetros | Tiempo de ajuste |
|---|---|---:|
| Regresión logística | `C = 1,0` | 5,1 s |
| KNN | `n_neighbors = 31`, `weights = uniform` | 0,8 s |
| Árbol de decisión | `max_depth = 7`, `min_samples_leaf = 20` | 0,6 s |
| Random Forest | `max_depth = 8`, `min_samples_leaf = 5` | 13,5 s |
| XGBoost | `max_depth = 3`, `learning_rate = 0,05` | 7,8 s |

### 7.3 Validación cruzada (5 particiones)

| Modelo | AP | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **0,668 ± 0,021** | 0,847 ± 0,011 | 0,791 ± 0,025 | 0,522 ± 0,011 | 0,629 ± 0,015 | 0,752 ± 0,008 |
| Random Forest | 0,662 ± 0,018 | **0,848 ± 0,009** | 0,793 ± 0,024 | 0,525 ± 0,016 | **0,632 ± 0,015** | 0,755 ± 0,012 |
| Regresión logística | 0,660 ± 0,020 | 0,846 ± 0,012 | **0,801 ± 0,038** | 0,517 ± 0,019 | 0,628 ± 0,023 | 0,748 ± 0,015 |
| Árbol de decisión | 0,620 ± 0,016 | 0,817 ± 0,010 | 0,780 ± 0,022 | 0,507 ± 0,032 | 0,614 ± 0,028 | 0,739 ± 0,025 |
| KNN | 0,620 ± 0,012 | 0,835 ± 0,009 | 0,583 ± 0,022 | **0,621 ± 0,023** | 0,601 ± 0,021 | **0,795 ± 0,011** |

Las métricas de recall, precision, F1 y accuracy corresponden al umbral 0,5.

> El modelo se ajustó y luego se volvió a validar sobre el mismo entrenamiento. Esto puede producir un leve optimismo en las cifras de validación; el conjunto de prueba confirma el resultado.

### 7.4 Umbral ajustado

Se eligió, para cada modelo, el umbral que maximiza F1 sobre las predicciones *out-of-fold*:

| Modelo | Umbral | F1 (OOF) | Recall (OOF) | Precision (OOF) |
|---|---:|---:|---:|---:|
| Regresión logística | 0,576 | 0,638 | 0,743 | 0,558 |
| XGBoost | 0,585 | 0,636 | 0,718 | 0,571 |
| Random Forest | 0,531 | 0,636 | 0,767 | 0,544 |
| KNN | 0,452 | 0,623 | 0,668 | 0,584 |
| Árbol de decisión | 0,525 | 0,615 | 0,761 | 0,516 |

### 7.5 Evaluación final en prueba

| Modelo | AP | ROC-AUC | Recall* | Precision* | F1* | Accuracy* |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **0,664** | **0,845** | 0,719 | 0,552 | 0,625 | 0,771 |
| Random Forest | 0,651 | 0,843 | **0,767** | 0,534 | **0,630** | 0,761 |
| Regresión logística | 0,633 | 0,842 | 0,725 | 0,539 | 0,618 | 0,762 |
| Árbol de decisión | 0,625 | 0,825 | **0,770** | 0,507 | 0,611 | 0,740 |
| KNN | 0,611 | 0,831 | 0,652 | **0,565** | 0,605 | **0,774** |

\* Umbral ajustado en validación.

**Lectura:**

- El orden por capacidad de discriminación es **XGBoost > Random Forest > regresión logística > árbol de decisión > KNN**.
- **KNN tiene la mayor accuracy (0,774) pero la menor AP (0,611) y el menor recall con umbral (0,652).** Es el ejemplo más claro de por qué la accuracy engaña.
- Los tres primeros modelos tienen un desempeño muy similar; las diferencias son pequeñas frente a la variabilidad entre particiones (±0,02).

### 7.6 Capacidad operativa: top 20 % de riesgo

Si el equipo de retención contacta al **20 % de la cartera con mayor probabilidad estimada**:

| Modelo | Contactados | Abandonos detectados | Cobertura | Precision en el grupo | Lift vs. azar |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | 282 | 195 | **52,1 %** | **69,1 %** | **2,61×** |
| Random Forest | 282 | 190 | 50,8 % | 67,4 % | 2,54× |
| KNN | 286 | 189 | 50,5 % | 66,1 % | 2,53× |
| Regresión logística | 282 | 186 | 49,7 % | 66,0 % | 2,49× |
| Árbol de decisión | 282 | 184 | 49,2 % | 65,2 % | 2,46× |

XGBoost detecta **195 de los 374 abandonos** del conjunto de prueba contactando a 282 clientes (20,0 % de la cartera): **2,6 veces más que una selección aleatoria**.

KNN mejora en este corte fijo porque concentra bien el extremo superior de riesgo, pero su AP global y su recall con umbral siguen siendo los más bajos.

### 7.7 Curvas y matrices de confusión

- Las curvas **ROC** de XGBoost, Random Forest y regresión logística son prácticamente indistinguibles (AUC 0,842–0,845); árbol y KNN quedan por debajo (0,825 y 0,831).
- Las curvas **precisión–recall** separan con más claridad a XGBoost (AP 0,664) de KNN (0,611), que es la métrica principal.
- En las matrices de confusión con umbral ajustado, Random Forest detecta 287 de los 374 abandonos (recall 0,767), con 250 falsos positivos; XGBoost detecta 269 (recall 0,719), con 218 falsos positivos.

> En esta ejecución, el umbral ajustado de XGBoost (0,585) elevó la precision de 0,525 a 0,552, pero redujo el recall de 0,789 a 0,719; el F1 en prueba no mejoró (0,625 frente a 0,630). El umbral elegido en validación no garantiza una mejora en datos nuevos y debe revisarse con el área de retención.

### 7.8 Interpretación de variables

**Random Forest: variables más importantes**

| Variable | Importancia |
|---|---:|
| `Contract_Month-to-month` | 0,152 |
| `tenure` | 0,120 |
| `TotalCharges` | 0,082 |
| `OnlineSecurity_No` | 0,074 |
| `Contract_Two year` | 0,070 |
| `TechSupport_No` | 0,057 |
| `MonthlyCharges` | 0,056 |
| `InternetService_Fiber optic` | 0,050 |

**Regresión logística: coeficientes condicionales**

| Mayor riesgo | Coeficiente | Menor riesgo | Coeficiente |
|---|---:|---|---:|
| `InternetService_Fiber optic` | +0,716 | `tenure` | −1,141 |
| `Contract_Month-to-month` | +0,663 | `Contract_Two year` | −0,766 |
| `TotalCharges` | +0,474 | `MonthlyCharges` | −0,677 |
| `StreamingMovies_Yes` | +0,283 | `InternetService_DSL` | −0,618 |
| | | `OnlineBackup_No internet service` | −0,270 |

**Precaución interpretativa:** los coeficientes son **condicionales**. `tenure`, `MonthlyCharges` y `TotalCharges` están fuertemente correlacionadas y `TotalCharges ≈ tenure × MonthlyCharges`; al controlar por las otras variables, el signo de `MonthlyCharges` puede invertirse (efecto de supresión). Además, las categorías `No internet service` de varios servicios son colineales entre sí. Por eso:

- Los coeficientes describen el comportamiento del modelo, **no efectos causales aislados**.
- La importancia de Random Forest se basa en impureza y favorece variables continuas o de alta cardinalidad; sirve como orientación, no como prueba de causalidad.
- Para explicar el negocio conviene complementar con análisis descriptivos y, idealmente, experimentos.

---

## 8. Conclusiones

### 8.1 ¿Cuál es el modelo más adecuado?

**XGBoost es la recomendación principal** para priorizar clientes según su riesgo de abandono:

- Mayor **average precision** en prueba (0,664).
- Mejor **ROC-AUC** (0,845).
- Mayor **cobertura de abandonos en el top 20 %** (52,1 %, lift 2,61×).
- Buen equilibrio recall/precision con umbral ajustado (0,719 / 0,552).

Sin embargo, la evidencia no muestra una superioridad contundente:

- **Random Forest** es la alternativa más cercana: AP 0,651, el mejor F1 con umbral ajustado (0,630) y un recall de 0,767. Si el objetivo es maximizar cobertura sin sacrificar demasiado la precision, es una elección defendible; el árbol de decisión logra un recall apenas mayor (0,770), pero con peor precision (0,507) y F1 (0,611).
- **Regresión logística** queda a solo 0,031 de AP del primer lugar y ofrece coeficientes interpretables, bajo costo de mantenimiento y facilidad para explicar cada alerta. Si la interpretabilidad es un requisito, es una opción muy razonable.
- **Árbol de decisión** y **KNN** quedan por debajo en AP y no se justifican frente a los tres anteriores para este problema.

Con una sola partición de prueba, las diferencias entre los tres primeros (0,013 a 0,031 de AP) son del mismo orden que la variabilidad observada en validación cruzada (±0,02). Por eso **no corresponde afirmar una superioridad estadísticamente concluyente**: XGBoost es el mejor candidato con la evidencia disponible, pero Random Forest y la regresión logística son alternativas válidas. Una validación repetida o temporal permitiría decidir con mayor solidez.

### 8.2 Limitaciones

1. **Datos transversales:** no permiten afirmar que el modelo predice el abandono “en los próximos 30 días”; estiman riesgo con base en el perfil actual.
2. **Asociación, no causalidad:** los coeficientes y las importancias describen el modelo, no el efecto de intervenir una variable.
3. **Desbalance moderado:** el 26,5 % de abandonos obliga a usar AP, recall y umbrales ajustados; la accuracy sola sería engañosa.
4. **Umbral dependiente del negocio:** el umbral óptimo cambia si cambia el costo de contactar o el valor de retener.
5. **Sin validación temporal:** un despliegue real debería entrenarse con datos históricos y evaluarse con clientes posteriores.
6. **Variables limitadas:** el conjunto no incluye interacciones con el servicio, reclamos, competencia ni contexto económico.
7. **Una sola partición de prueba:** los resultados tienen variabilidad; conviene validación repetida o temporal antes de decidir.

### 8.3 Recomendaciones

- Usar el modelo como **priorizador de campañas de retención**, no como decisión automática de cancelación.
- Revisar el umbral con el área de retención según su capacidad operativa: con 20 % de la cartera se cubre algo más de la mitad de los abandonos.
- Reentrenar periódicamente y monitorear la deriva de los datos.
- Complementar con experimentos o análisis causales antes de concluir que una acción evita el abandono.

---

## 9. Estructura del notebook

`telco_churn.ipynb` contiene únicamente código comentado, sin celdas de texto. Los bloques son:

| Bloque | Contenido |
|---:|---|
| 1 | Configuración e importaciones |
| 2 | Carga y revisión inicial |
| 3 | Depuración |
| 4 | Partición estratificada |
| 5 | Análisis exploratorio |
| 6 | Preparación de variables |
| 7 | Modelos y grillas |
| 8 | Línea base |
| 9 | Ajuste y validación cruzada |
| 10 | Ajuste del umbral |
| 11 | Evaluación en prueba |
| 12 | Capacidad operativa (top 20 %) |
| 13 | Curvas ROC y precisión–recall |
| 14 | Matrices de confusión |
| 15 | Coeficientes de la regresión logística |
| 16 | Importancia de variables en Random Forest |
| 17 | Ranking final y recomendación |

---

## 10. Referencias

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21–27.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). Wiley.
- IBM. (2015). *Telco Customer Churn* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
