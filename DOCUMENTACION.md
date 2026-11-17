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

**Resultado principal:** el modelo elegido fue **Random Forest**, con un **F1 de 0,624** sobre clientes que nunca vio durante el entrenamiento. Detecta al **78 % de los clientes que abandonan**, aunque solo la mitad de sus alertas son correctas.

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

---

## 4. Los cinco modelos

| Modelo | En una frase |
|---|---|
| **Regresión logística** | Suma las variables con pesos. Es el más fácil de interpretar. |
| **KNN** | Mira a los 31 clientes más parecidos y decide por mayoría. |
| **Árbol de decisión** | Crea reglas: “si el contrato es mensual y la antigüedad es baja…”. |
| **Random Forest** | Combina 300 árboles distintos y promedia sus respuestas. |
| **XGBoost** | Construye árboles uno tras otro, corrigiendo los errores del anterior. |

**Configuración:** usamos valores razonables y fijos para cada modelo, sin búsqueda automática de hiperparámetros. La meta es **entender y comparar**, no exprimir el último decimal.

**Clases desbalanceadas:** como solo el 26,5 % abandona, damos más peso a los abandonos durante el entrenamiento (`class_weight="balanced"` en tres modelos, `scale_pos_weight` en XGBoost).

---

## 5. Las tres métricas que importan

Para un problema de abandono, estas son las preguntas relevantes:

| Métrica | Pregunta que responde | Ejemplo |
|---|---|---|
| **Recall** | De los que se fueron, ¿a cuántos detectamos? | 0,78 = detectamos 78 de cada 100 |
| **Precision** | De los que alertamos, ¿cuántos se fueron de verdad? | 0,52 = la mitad de las alertas son correctas |
| **F1** | Equilibrio entre las dos anteriores | Media armónica de recall y precision |

**Accuracy** se muestra solo como referencia, porque puede ser alta sin detectar ni un solo abandono.

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

Además, los abandonos se concentran en los **primeros meses** de antigüedad.

**Importante:** esto son asociaciones, no causas. Que quienes tienen contrato mensual abandonen más no significa que cambiar el contrato lo evite.

---

## 8. Conclusiones

### 8.1 ¿Cuál es el modelo más adecuado?

**Random Forest**, para priorizar clientes en una campaña de retención:

- Mejor **F1 en validación** (0,627) y confirmado en prueba (0,624).
- Detecta al **78 % de los clientes que abandonan**.
- La mitad de sus alertas son correctas, un costo razonable frente a perder clientes.

**Alternativas cercanas:**

- **Regresión logística:** F1 0,624 en validación, prácticamente empatada. Es la más fácil de explicar: sus coeficientes indican qué variables empujan el riesgo hacia arriba o hacia abajo. Si el negocio pide entender cada alerta, es una excelente opción.
- **XGBoost:** quedó tercero (F1 0,611) con esta configuración fija. Es un modelo potente, pero su ventaja suele aparecer al ajustar hiperparámetros. Aquí primó la simplicidad.
- **KNN:** la mayor accuracy (0,793), pero el peor recall (0,561). Otro ejemplo de que la accuracy engaña.
- **Árbol de decisión:** el mayor recall (0,799) con la peor precision (0,493). Generaría demasiadas falsas alarmas.

**Sin embargo**, las diferencias entre los tres primeros son pequeñas (F1 0,624–0,627). No hicimos una prueba estadística, así que **no puede afirmarse que Random Forest sea superior de forma concluyente**. Es el mejor candidato con esta evidencia.

### 8.2 Limitaciones

1. **Datos de un solo momento:** no hay fechas, así que no podemos predecir “en los próximos 30 días”; estimamos el riesgo con el perfil actual.
2. **Asociación, no causalidad.**
3. **Umbral fijo en 0,5:** el óptimo depende de cuántos clientes pueda contactar la empresa.
4. **Una sola división de datos:** con otra semilla los números podrían variar ligeramente.
5. **Variables limitadas:** faltan reclamos, competencia y contexto económico.

### 8.3 Recomendaciones

- Usar el modelo para **ordenar** clientes, no para decidir automáticamente.
- Ajustar el umbral según la capacidad real de contacto.
- Reentrenar con datos nuevos y vigilar que los patrones no cambien.
- Medir con una campaña real si la retención funciona: eso sí sería causal.

---

## 9. Estructura del notebook

`telco_churn.ipynb` contiene **solo código comentado**, sin celdas de texto: 20 celdas.

| Celda | Contenido |
|---:|---|
| 1 | Importaciones y configuración |
| 2 | Cargar los datos |
| 3 | Revisar la variable objetivo |
| 4 | Limpiar los datos |
| 5 | Dividir en entrenamiento / validación / prueba |
| 6 | Explorar los datos |
| 7 | Gráfico: abandono, contrato y antigüedad |
| 8 | Preparar las variables |
| 9 | Referencia: “siempre no abandona” |
| 10–14 | Los cinco modelos, uno por celda |
| 15 | Comparar en validación |
| 16 | Gráfico: comparación de F1 |
| 17 | Elegir el modelo |
| 18 | Evaluación final en prueba |
| 19 | Matriz de confusión |
| 20 | Resumen final |

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

El archivo `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` debe estar en la carpeta `data/`.

> El kernel es **Python 3.15.2**. El servidor de JupyterLab arranca con 3.14 porque `PyYAML` todavía no tiene versión binaria para 3.15. Los modelos y las métricas sí se ejecutan en 3.15.

---

## 11. Referencias

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794.
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21–27.
- IBM. (2015). *Telco Customer Churn*. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
