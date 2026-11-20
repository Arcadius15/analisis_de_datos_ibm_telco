# Predicción de abandono de clientes en telecomunicaciones

**Tipo de trabajo:** clasificación supervisada  
**Datos:** IBM Telco Customer Churn — 7.043 clientes  
**Modelos comparados:** regresión logística, KNN, árbol de decisión, Random Forest y XGBoost  
**Entorno:** Python 3.15.2 · pandas 3.1.1 · scikit-learn 1.9.1 · XGBoost 3.4.1

---

## 1. ¿Qué hace este trabajo?

Una empresa de telecomunicaciones pierde clientes cada mes. Conseguir un cliente nuevo cuesta más que retener uno existente, así que conviene **saber quién tiene riesgo de irse antes de que se vaya**.

Construimos un modelo que, a partir de datos del cliente (antigüedad, contrato, servicios, facturación), estima si tiene **alta o baja probabilidad de abandonar**. Con esa lista, el equipo de retención puede llamar primero a los clientes con mayor riesgo.

**Qué NO hace:** no demuestra que llamar a un cliente evite que se vaya. Solo identifica patrones en datos históricos.

**Resultado principal:** el modelo elegido fue **Random Forest**, con un **F1 de 0,624** sobre clientes que nunca vio durante el entrenamiento. Detecta al **78 % de los clientes que abandonan**, aunque solo la mitad de sus alertas son correctas. Su **ROC-AUC es 0,842** y cumple **las cuatro metas** del proyecto (ROC-AUC ≥ 0,80, recall ≥ 0,75, F1 ≥ 0,60 y accuracy mejor que la referencia).

---

## 2. Los datos

| Concepto | Valor |
|---|---|
| Clientes | 7.043 |
| Columnas | 21 |
| Abandonan | 1.869 (26,5 %) |
| No abandonan | 5.174 (73,5 %) |

Las variables se agrupan así:

- **Datos personales:** género, adulto mayor, pareja, dependientes.
- **Servicios:** teléfono, internet, seguridad, soporte técnico, streaming.
- **Contrato y pago:** tipo de contrato, factura electrónica, método de pago.
- **Facturación:** cargo mensual, cargo acumulado, antigüedad en meses.
- **Objetivo:** `Churn` (Yes = abandonó).

### Dos detalles importantes

1. **`customerID` se elimina.** Es un número de identificación: no describe al cliente y el modelo no debe aprenderlo.
2. **`TotalCharges` llega como texto** porque 11 registros están vacíos. Esos 11 clientes tienen **antigüedad 0**: son clientes nuevos que todavía no acumulan facturación. Se rellenan con 0.

**El 73,5 % no abandona.** Esto significa que un modelo que siempre diga “no abandona” acertaría 3 de cada 4 casos. Por eso **no usamos la exactitud (accuracy) como medida principal**.

**Fuente:** IBM. (2015). *Telco Customer Churn*. Kaggle.

---

## 3. Cómo dividimos los datos

Antes de tocar nada, separamos los clientes en tres grupos:

| Grupo | Clientes | Para qué |
|---|---:|---|
| Entrenamiento | 4.225 | El modelo aprende |
| Validación | 1.409 | Comparamos los 5 modelos y elegimos uno |
| Prueba | 1.409 | Comprobamos el resultado final, una sola vez |

Los tres grupos tienen la misma proporción de abandonos (26,5 %). Esto se consigue con `stratify`.

**¿Por qué tres grupos y no dos?** Si eligiéramos el modelo mirando el grupo de prueba, estaríamos haciendo trampa: el resultado se vería mejor de lo que realmente es. La regla es simple:

> Elegir con validación. Comprobar con prueba. No cambiar nada después.

**Una precisión sobre «una sola vez».** Las ampliaciones calculan varias métricas sobre prueba (ROC-AUC, comparación de modelos ajustados y cumplimiento de metas). Todas son **descriptivas**: se reportan, no se usan para elegir ni para cambiar hiperparámetros. La decisión sigue tomándose en validación, y por eso la estimación final conserva su valor. Lo que sí invalidaría el resultado es mirar prueba y después volver a ajustar el modelo.

---

## 4. Los cinco modelos

| Modelo | En una frase |
|---|---|
| **Regresión logística** | Suma las variables con pesos. Es el más fácil de interpretar. |
| **KNN** | Mira a los 31 clientes más parecidos y decide por mayoría. |
| **Árbol de decisión** | Crea reglas: “si el contrato es mensual y la antigüedad es baja…”. |
| **Random Forest** | Combina 300 árboles distintos y promedia sus respuestas. |
| **XGBoost** | Construye árboles uno tras otro, corrigiendo los errores del anterior. |

**Configuración base:** usamos valores razonables y fijos para cada modelo. La meta es **entender y comparar**, no exprimir el último decimal.

**Ajuste complementario:** en la **Ampliación 4** se busca un mejor juego de hiperparámetros para regresión logística, Random Forest y XGBoost con `GridSearchCV` (5 particiones, ROC-AUC, solo entrenamiento). La elección del modelo de la **sección 15 del notebook** **no cambia** con ese ajuste; sirve para comprobar cuánto margen de mejora había. Los resultados se comentan en 6.7.

**Clases desbalanceadas:** como solo el 26,5 % abandona, damos más peso a los abandonos durante el entrenamiento (`class_weight="balanced"` en tres modelos, `scale_pos_weight` en XGBoost).

---

## 5. Las cuatro métricas que importan

Para un problema de abandono, estas son las preguntas relevantes:

| Métrica | Pregunta que responde | Ejemplo |
|---|---|---|
| **Recall** | De los que se fueron, ¿a cuántos detectamos? | 0,78 = detectamos 78 de cada 100 |
| **Precision** | De los que alertamos, ¿cuántos se fueron de verdad? | 0,52 = la mitad de las alertas son correctas |
| **F1** | Equilibrio entre las dos anteriores | Media armónica de recall y precision |
| **ROC-AUC** | ¿El modelo ordena bien el riesgo? | 0,84 = separa bastante bien a quien se va de quien se queda |

**Accuracy** se muestra solo como referencia, porque puede ser alta sin detectar ni un solo abandono.

**Qué aporta el ROC-AUC.** Recall, precision y F1 dependen de dónde pongamos el umbral de decisión (aquí, 0,5). El ROC-AUC no: mide la capacidad de **ordenar** a los clientes por riesgo, y es la métrica que fija la rúbrica. Un valor de 0,5 equivale a adivinar al azar; 1,0 sería separación perfecta.

**El intercambio:** si bajamos el umbral, detectamos más abandonos, pero también generamos más falsas alarmas. Si lo subimos, lo contrario. No existe un modelo perfecto en ambas.

---

## 6. Resultados

### 6.1 Referencia

Un modelo que siempre responde “no abandona” obtiene:

- **Accuracy:** 0,735
- **Recall:** 0,000

Es decir: acierta mucho y no sirve para nada. Esta es la razón por la que no elegimos por accuracy.

### 6.2 Comparación en validación

| Modelo | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|
| **Random Forest** | 0,781 | 0,523 | **0,627** | 0,753 |
| Regresión logística | 0,789 | 0,517 | 0,624 | 0,748 |
| XGBoost | 0,775 | 0,503 | 0,611 | 0,737 |
| Árbol de decisión | 0,799 | 0,493 | 0,610 | 0,729 |
| KNN | 0,561 | 0,621 | 0,590 | 0,793 |

**Modelo elegido: Random Forest**, por tener el mayor F1. Esta decisión se tomó **sin mirar el conjunto de prueba**.

### 6.3 Evaluación final en prueba

Con el modelo ya elegido:

| Métrica | Valor |
|---|---:|
| Recall | **0,778** |
| Precision | 0,522 |
| F1 | **0,624** |
| Accuracy | 0,752 |

El resultado en prueba (F1 0,624) es casi idéntico al de validación (0,627). Eso indica que el modelo **no se sobreajustó** y generaliza de forma estable.

### 6.4 Matriz de confusión

Sobre 1.409 clientes de prueba, de los cuales 374 abandonaron:

| | Predijo “no abandona” | Predijo “abandona” |
|---|---:|---:|
| **No abandonó** (1.035) | 768 | 267 |
| **Abandonó** (374) | 83 | 291 |

Lectura:

- **291 abandonos detectados** de 374 (recall 0,778).
- **83 abandonos no detectados.**
- **267 falsas alarmas:** clientes que no iban a irse.

En una campaña real, esas 267 llamadas innecesarias son el costo de detectar a 291 clientes en riesgo. Si el equipo solo puede hacer 300 llamadas, el umbral debería subirse.

### 6.5 ROC-AUC

El ROC-AUC mide la capacidad de ordenar el riesgo sin depender del umbral. En validación y en prueba:

| Modelo | ROC-AUC validación | ROC-AUC prueba |
|---|---:|---:|
| **Random Forest** | **0,837** | **0,842** |
| XGBoost | 0,836 | 0,839 |
| Regresión logística | 0,836 | 0,843 |
| Árbol de decisión | 0,824 | 0,832 |
| KNN | 0,824 | 0,830 |

Dos lecturas:

1. **Todos superan 0,80**, que era la meta. La separación de riesgo es buena incluso en los modelos flojos en F1.
2. **El orden cambia respecto a F1.** KNN tiene el ROC-AUC más bajo en prueba (0,830) y también el peor F1; la regresión logística es la mejor en prueba por una milésima sobre Random Forest (0,843 frente a 0,842). Las diferencias entre los tres primeros (0,839–0,843) son tan pequeñas que el ROC-AUC no sirve para desempatar: para eso está F1, que además es la métrica de la decisión.

**Conclusión:** el ROC-AUC confirma que los modelos son útiles, no cuál elegir.

### 6.6 Validación cruzada

Una sola partición puede dar un resultado por suerte. La validación cruzada repite el entrenamiento y la evaluación 5 veces sobre el 80 % de los datos (entrenamiento + validación); **la prueba no participa**.

| Modelo | Recall | Precision | F1 | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | 0,794 ± 0,019 | 0,529 ± 0,016 | **0,635 ± 0,014** | 0,847 ± 0,009 | 0,758 ± 0,012 |
| XGBoost | 0,799 ± 0,023 | 0,524 ± 0,013 | 0,633 ± 0,017 | 0,846 ± 0,011 | 0,754 ± 0,011 |
| Regresión logística | 0,801 ± 0,038 | 0,517 ± 0,019 | 0,628 ± 0,023 | 0,846 ± 0,012 | 0,748 ± 0,015 |
| Árbol de decisión | 0,777 ± 0,036 | 0,511 ± 0,011 | 0,616 ± 0,011 | 0,829 ± 0,014 | 0,743 ± 0,008 |
| KNN | 0,583 ± 0,022 | 0,621 ± 0,023 | 0,601 ± 0,021 | 0,835 ± 0,009 | 0,795 ± 0,011 |

Los números van con **± la desviación entre particiones**, que es la medida de cuánto depende el resultado de la división concreta.

Ideas importantes:

- **El podio se mantiene arriba y abajo:** Random Forest primero (F1 0,635) y KNN último (0,601), igual que en la comparación de validación. Lo único que cambia es el segundo puesto: XGBoost (0,633) adelanta a la regresión logística (0,628). La distancia entre Random Forest y XGBoost es de dos milésimas, **menor que la variación entre particiones**, así que no hay un ganador claro: la elección de Random Forest es defendible, no concluyente.
- **La regresión logística no es inferior:** su F1 medio (0,628) está dentro del rango de Random Forest y es la más variable (± 0,023).
- **KNN se confirma:** mejor accuracy (0,795) y peor recall (0,583). El patrón de la sección 6.2 no era casualidad.

### 6.7 Ajuste de hiperparámetros

Se probaron combinaciones con `GridSearchCV` (5 particiones, ROC-AUC) **solo con el conjunto de entrenamiento**. Los mejores valores:

| Modelo | Mejor configuración | ROC-AUC en CV |
|---|---|---:|
| Regresión logística | `C = 10` (menos regularización) | 0,848 |
| Random Forest | `max_depth = 10`, `min_samples_leaf = 20` | 0,850 |
| XGBoost | `n_estimators = 200`, `learning_rate = 0,03`, `max_depth = 3` | 0,847 |

Comparación con los valores fijos:

| Modelo | Versión | F1 val | ROC-AUC val | Recall prueba | F1 prueba |
|---|---|---:|---:|---:|---:|
| Regresión logística | fijos | 0,624 | 0,836 | 0,783 | 0,614 |
| | ajustado | 0,620 | 0,837 | 0,786 | 0,616 |
| **Random Forest** | **fijos** | **0,627** | 0,837 | 0,778 | 0,624 |
| | ajustado | 0,621 | 0,838 | 0,797 | 0,625 |
| XGBoost | fijos | 0,611 | 0,836 | 0,791 | 0,623 |
| | ajustado | 0,617 | 0,839 | 0,805 | 0,623 |

**El ajuste no cambia la historia.** Las mejoras en F1 de validación son de centésimas, y en **dos de los tres modelos** el ajustado queda **peor** que el de valores fijos: Random Forest 0,627 → 0,621 y regresión logística 0,624 → 0,620. Solo XGBoost mejora (0,611 → 0,617), seis milésimas. Lo que sí sube es el recall en prueba de Random Forest (0,778 → 0,797): detecta más abandonos, pero el F1 apenas cambia (0,624 → 0,625) porque también aumentan las falsas alarmas. Con 7.043 clientes, ese margen está dentro del ruido.

Por eso la decisión de la **sección 15 del notebook** **no se modifica**: se mantiene el modelo con valores fijos, que es más simple de explicar y rinde igual.

### 6.8 Cumplimiento de las metas

Metas del proyecto: ROC-AUC ≥ 0,80 · recall ≥ 0,75 · F1 ≥ 0,60 · accuracy superior a la referencia (0,735). Se calculan sobre prueba **después** de cerrar el análisis, solo para reportar:

| Modelo | ROC-AUC | Recall | F1 | Accuracy | ¿4 metas? |
|---|---:|---:|---:|---:|:---:|
| **Random Forest (elegido)** | 0,842 | 0,778 | 0,624 | 0,752 | **Sí** |
| Regresión logística | 0,843 | 0,783 | 0,614 | 0,739 | Sí |
| XGBoost | 0,839 | 0,791 | 0,623 | 0,745 | Sí |
| Árbol de decisión | 0,832 | 0,783 | 0,614 | 0,739 | Sí |
| KNN | 0,830 | 0,575 | 0,583 | 0,781 | **No** |
| Regresión logística (ajustada) | 0,842 | 0,786 | 0,616 | 0,740 | Sí |
| Random Forest (ajustado) | 0,844 | 0,797 | 0,625 | 0,746 | Sí |
| XGBoost (ajustado) | 0,844 | 0,805 | 0,623 | 0,741 | Sí |

**Siete de los ocho candidatos cumplen las cuatro metas**; el que no, KNN, falla por recall (0,575 < 0,75) y F1 (0,583 < 0,60). **Esta tabla es descriptiva:** comprueba las metas una vez terminado el análisis y no se usó para elegir modelo. Si se usara para elegir, se estaría eligiendo con el conjunto de prueba y el resultado dejaría de ser una estimación honesta.

---

## 7. ¿Qué aprendimos de los datos?

Solo con el grupo de entrenamiento:

| Variable | Categoría | Tasa de abandono |
|---|---|---:|
| Contrato | Mes a mes | **43,1 %** |
| | Un año | 10,8 % |
| | Dos años | 2,5 % |
| Internet | Fibra óptica | **42,3 %** |
| | DSL | 18,7 % |
| | Sin internet | 7,3 % |
| Pago | Cheque electrónico | **45,6 %** |
| | Transferencia automática | 16,2 % |
| | Tarjeta automática | 15,1 % |

Además, los abandonos se concentran en los **primeros meses** de antigüedad. Con el EDA ampliado (Ampliación 1) se ve el detalle:

| Variable | Categoría | Tasa de abandono |
|---|---|---:|
| Antigüedad | 0–12 meses | **47,3 %** |
| | 13–24 meses | 29,3 % |
| | 25–48 meses | 19,8 % |
| | 49–72 meses | 9,6 % |
| Soporte técnico | Sin soporte | **41,8 %** |
| | Sin internet | 7,3 % |
| | Con soporte | 15,0 % |
| Seguridad en línea | Sin seguridad | **42,0 %** |
| | Sin internet | 7,3 % |
| | Con seguridad | 14,1 % |
| Factura electrónica | Sí | **33,5 %** |
| | No | 16,2 % |
| Adulto mayor | Sí | **40,8 %** |
| | No | 23,7 % |

Dos medianas que resumen el perfil de quien se va: **10 meses** de antigüedad frente a **37** de quien se queda, y **79,75** de cargo mensual frente a **64,38**. La correlación con el abandono es −0,353 para `tenure` y +0,196 para `MonthlyCharges`: la antigüedad pesa más que el precio.

El **contrato mensual** es el factor más importante del modelo, y no es casualidad que coincida con la categoría de mayor tasa de abandono (43,1 %). Pero sigue siendo una asociación: quien firma mes a mes puede estar ya decidido a irse y por eso no se compromete.

Y otra vez: **esto son asociaciones, no causas.** Que quienes tienen contrato mensual abandonen más no significa que cambiar el contrato lo evite.

---

## 8. Conclusiones

### 8.1 ¿Cuál es el modelo más adecuado?

**Random Forest**, para priorizar clientes en una campaña de retención:

- Mejor **F1 en validación** (0,627) y confirmado en prueba (0,624).
- La validación cruzada lo respalda: **F1 0,635 ± 0,014**, el mejor promedio de los cinco.
- **ROC-AUC 0,842** en prueba, por encima de la meta de 0,80.
- Detecta al **78 % de los clientes que abandonan**.
- **Cumple las cuatro metas** del proyecto.
- La mitad de sus alertas son correctas, un costo razonable frente a perder clientes.

**Alternativas cercanas:**

- **Regresión logística:** F1 0,624 en validación, prácticamente empatada (0,628 ± 0,023 en validación cruzada). Es la más fácil de explicar: sus coeficientes indican qué variables empujan el riesgo hacia arriba o hacia abajo. Si el negocio pide entender cada alerta, es una excelente opción.
- **XGBoost:** quedó tercero con la configuración fija (F1 0,611), pero en validación cruzada sube a **0,633 ± 0,017**, a dos milésimas de Random Forest. Al ajustar hiperparámetros no mejora de forma apreciable (F1 prueba 0,623 → 0,623).
- **KNN:** la mayor accuracy (0,793) y el peor recall (0,561). Es el único que **no cumple las metas**. Otro ejemplo de que la accuracy engaña.
- **Árbol de decisión:** el mayor recall (0,799) con la peor precision (0,493). Generaría demasiadas falsas alarmas.

**Sin embargo**, las diferencias entre los tres primeros son pequeñas y **la validación cruzada lo confirma**: Random Forest (0,635 ± 0,014) y XGBoost (0,633 ± 0,017) se solapan dentro de su variación. No hicimos una prueba estadística formal, así que **no puede afirmarse que Random Forest sea superior de forma concluyente**. Es el mejor candidato con esta evidencia, y las tres alternativas razonables cumplen las metas igual.

### 8.2 Limitaciones

1. **Datos de un solo momento:** no hay fechas, así que no podemos predecir “en los próximos 30 días”; estimamos el riesgo con el perfil actual.
2. **Asociación, no causalidad.**
3. **Umbral fijo en 0,5:** el óptimo depende de cuántos clientes pueda contactar la empresa.
4. **División única con semilla fija:** la validación cruzada reduce esta limitación (el orden se mantiene en las 5 particiones), pero sigue siendo un solo conjunto de prueba.
5. **Variables limitadas:** faltan reclamos, competencia y contexto económico.
6. **Las metas se comprueban, no se optimizan:** la tabla de la sección 6.8 se calculó después de cerrar el análisis. Optimizar contra ella sería usar el conjunto de prueba para elegir y volvería el resultado optimista.

### 8.3 Recomendaciones

- Usar el modelo para **ordenar** clientes, no para decidir automáticamente.
- Ajustar el umbral según la capacidad real de contacto.
- Reentrenar con datos nuevos y vigilar que los patrones no cambien.
- Medir con una campaña real si la retención funciona: eso sí sería causal.

---

## 9. Estructura del notebook

`telco_churn.ipynb` contiene **31 celdas de código comentado** (las 20 de la versión base más las ampliaciones) y **8 celdas de texto** que sirven de guía. Las secciones numeradas 1–17 conservan la lógica y los resultados originales; las ampliaciones se marcan con su propio título.

| Celda | Contenido |
|---:|---|
| 1 | Importaciones y configuración (sección base 1) |
| 2 | Texto: cómo está organizado el notebook |
| 3 | Ampliación 0: descarga del CSV e importaciones adicionales |
| 4 | Sección base 2: cargar los datos |
| 5 | Sección base 3: revisar la variable objetivo |
| 6 | Sección base 4: limpiar los datos |
| 7 | Sección base 5: dividir en entrenamiento / validación / prueba |
| 8 | Sección base 6: explorar los datos |
| 9 | Figura 1: abandono, contrato y antigüedad |
| 10 | Texto: Ampliación 1 |
| 11 | Ampliación 1: EDA ampliado |
| 12 | Figura 1b: antigüedad, soporte y cargo mensual |
| 13 | Sección base 7: preparar las variables |
| 14 | Sección base 8: referencia “siempre no abandona” |
| 15–19 | Secciones base 9–13: los cinco modelos, uno por celda |
| 20 | Sección base 14: comparar en validación |
| 21 | Figura 2: comparación de F1 |
| 22 | Sección base 15: elegir el modelo |
| 23 | Sección base 16: evaluación final en prueba |
| 24 | Figura 3: matriz de confusión |
| 25 | Sección base 17: resumen final |
| 26 | Texto: Ampliación 2 |
| 27 | Ampliación 2: ROC-AUC en validación y prueba |
| 28 | Figura 4: curvas ROC |
| 29 | Texto: Ampliación 3 |
| 30 | Ampliación 3: validación cruzada de 5 particiones |
| 31 | Texto: Ampliación 4 |
| 32 | Ampliación 4: GridSearchCV |
| 33 | Ampliación 4: tabla fijos vs. ajustados |
| 34 | Texto: Ampliación 5 |
| 35 | Ampliación 5: cumplimiento de metas en prueba |
| 36 | Texto: Ampliación 6 |
| 37 | Ampliación 6 y Figura 5: importancia de variables y coeficientes |
| 38 | Texto: Ampliación 7 |
| 39 | Ampliación 7: resumen con veredicto de metas |

Las celdas de texto no llevan número de sección: solo explican qué hace el bloque que sigue, para que el notebook se pueda leer de arriba a abajo sin abrir la documentación.

---

## 10. Cómo ejecutarlo

```bash
# 1) Sincronizar el entorno
uv sync

# 2) Registrar el kernel de Python 3.15
uv run python -m ipykernel install --user \
    --name telco-churn \
    --display-name "Python 3.15 (Telco Churn)"

# 3) Abrir JupyterLab
uvx --python 3.14 --with jupyterlab jupyter lab
```

El archivo `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` se **descarga automáticamente** la primera vez que se ejecuta el notebook si no está en la carpeta `data/`. Si ya lo tienes, se usa tal cual (se comprobó que el archivo local es idéntico al del repositorio de IBM).

Para ejecutar todo de una vez sin abrir la interfaz:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=telco-churn telco_churn.ipynb
```

Tarda unos **2 minutos**, casi todo en el `GridSearchCV` de XGBoost (36 combinaciones × 5 particiones).

> El kernel es **Python 3.15.2**. El servidor de JupyterLab arranca con 3.14 porque `PyYAML` todavía no tiene versión binaria para 3.15. Los modelos y las métricas sí se ejecutan en 3.15.

---

## 11. Referencias

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794.
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21–27.
- IBM. (2015). *Telco Customer Churn*. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
