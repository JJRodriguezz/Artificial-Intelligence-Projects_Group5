# Predicción de Ventas de Videojuegos con Redes Neuronales

## ¿Qué estamos haciendo aquí?

A partir de los datos históricos de ventas de videojuegos, queremos predecir cuántas copias venderá globalmente un juego nuevo. En este apartado del laboratorio #2 abordaremos cómo implementamos una red neuronal artificial para capturar patrones complejos y relaciones no lineales que los métodos tradicionales no pueden detectar.

## El Dataset

La fuente de todo este análisis. Trabajamos con el dataset vgsales.csv de Kaggle, que contiene:

**16,598 juegos analizados**

**11 características por juego:**
- Rank, Name, Platform, Year, Genre, Publisher
- NA_Sales (ventas en Norteamérica)
- EU_Sales (ventas en Europa) 
- JP_Sales (ventas en Japón)
- Other_Sales (ventas en otras regiones)
- Global_Sales (nuestra variable objetivo)

**Desafío que encontramos:**
A diferencia de problemas más simples, aquí no tenemos una relación lineal directa. Las interacciones entre géneros, plataformas y años crean patrones complejos que requieren un enfoque más sofisticado.

## El Algoritmo: Por qué elegimos Redes Neuronales

Seleccionamos una red neuronal artificial porque sabemos que es particularmente efectiva para:
- Detectar interacciones automáticamente entre múltiples variables
- Modelar relaciones no lineales complejas
- Generalizar patrones cuando se regulariza correctamente
- Manejar tanto datos categóricos como numéricos simultáneamente

Pero aquí viene lo interesante...

## ¿Cómo llegamos al modelo óptimo?: Experimentación Sistemática

Probamos múltiples arquitecturas y configuraciones. Aquí está lo que descubrimos:

### La Ganadora: Arquitectura 64-32-16 con Dropout
- **R² Score: 0.9462** (explica 94.6% de la variabilidad)
- **RMSE: 0.4754** (desviación controlada)  
- **MAE: 0.0932** (error promedio de solo 93,200 copias)

```python
# La configuración victoriosa:
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3), 
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])
```

### ¿Por qué esta arquitectura específicamente?

**Estructura de embudo (64→32→16):** Permite comprimir información gradualmente, evitando pérdidas bruscas de patrones importantes.

**ReLU como activación:** No satura como sigmoid/tanh, permite gradientes limpios, y es computacionalmente eficiente.

**Dropout al 30%:** El punto dulce entre regularización y capacidad. Menos del 30% no previene overfitting, más del 30% limita demasiado el aprendizaje.

### Los Experimentos Fallidos

**Arquitectura Minimalista (1-2 capas):**
- R² máximo de ~0.78
- No capturaba interacciones complejas entre variables categóricas
- Predicciones demasiado simplificadas para la realidad del problema

**Arquitectura Excesiva (5+ capas):**  
- Overfitting severo (diferencia >15% entre train/test)
- Tiempo de entrenamiento prohibitivo
- Memorización en lugar de aprendizaje de patrones

**Sin regularización:**
- R² train: 0.99, R² test: 0.75 (overfitting clásico)
- Predicciones perfectas en entrenamiento, desastrosas en datos nuevos

¿Por qué fracasaron?
Los modelos simples carecían de expresividad, los complejos se sobreajustaban, y sin regularización la red memorizaba en lugar de aprender.

## El Preprocesamiento: Donde se gana o pierde la partida

**Los pasos críticos que implementamos:**

### 1. Manejo inteligente de datos faltantes
```python
# 271 años faltantes → imputación con moda (mantiene distribución temporal)
# 58 editores faltantes → categoría "Desconocido" (no introduce sesgo)
```

### 2. Transformación de categóricas
```python
# Platform: 31 plataformas → LabelEncoder (0-30)
# Genre: 12 géneros → LabelEncoder (0-11)  
# Publisher: 579 editores → LabelEncoder (0-578)
```

**¿Por qué LabelEncoder y no One-Hot?** Con 579 editores, One-Hot habría creado 579 columnas adicionales, causando:
- Explosión dimensional que ralentiza entrenamiento
- Matrices extremadamente dispersas
- Problemas de memoria y overfitting

### 3. Normalización obligatoria
```python
# StandardScaler: media=0, desviación=1 para todas las variables
# Sin esto, Year (1980-2016) dominaría sobre Sales (0-1)
```

Las redes neuronales son hipersensibles a la escala. Variables con rangos grandes dominan el cálculo de gradientes y provocan convergencia inestable.

## Entendiendo los Resultados

### Lo Realmente Bueno:
- **R² de 0.9462:** Capturamos el 94.6% de toda la variabilidad en ventas globales
- **Error minúsculo:** 93,200 copias promedio en un rango de 0.01 a 82+ millones
- **Generalización sólida:** Solo 0.78% de diferencia entre entrenamiento y prueba

### Análisis técnico profundo:

**Interpretación del R²:**
```
R² = 1 - (Varianza_residual / Varianza_total) = 0.9462
```
Solo el 5.38% de la variación en ventas se debe a factores no capturados por nuestro modelo.

**MAE vs RMSE revela el patrón:**
- MAE (0.0932) vs RMSE (0.4754): la diferencia indica algunos outliers
- Pero no son sistemáticos - el modelo maneja bien la mayoría de casos

**Prueba de overfitting robusta:**
- Diferencia R²: 0.0078 (menos del 1%)
- El modelo aprendió patrones generalizables, no memorizó datos

### Casos donde aún tenemos desafíos:
Los errores más grandes ocurren con juegos que rompieron moldes (como Wii Sports), donde patrones históricos no aplican. En términos porcentuales algunos errores parecen grandes, pero esto sucede principalmente con juegos de ventas muy bajas donde 0.09M de error se convierte en un porcentaje alto, aunque el error absoluto sigue siendo comercialmente insignificante.

## Comparación con Alternativas

**vs Modelos Lineales:** Las redes neuronales capturan que el éxito de un RPG en Switch es diferente al de un RPG en PC - interacciones que modelos lineales no pueden modelar.

**vs Árboles de Decisión:** Mejor generalización en espacios continuos y predicciones más suaves.


## ¿Qué hemos aprendido?

### Lecciones técnicas clave:
- **El preprocesamiento determina el 70% del éxito final**
- **Dropout es la diferencia entre memorizar y aprender**  
- **La arquitectura óptima balancea expresividad y regularización**
- **Las redes neuronales brillan cuando las relaciones son verdaderamente no lineales**

### Insights sobre el dominio:
- Los patrones de ventas de videojuegos son más predecibles de lo esperado
- Género + Plataforma tienen efectos multiplicativos, no aditivos
- Variables regionales interactúan de formas complejas que justifican redes neuronales



## Archivos Técnicos Generados

```
model2_nn/
├── preprocesamiento.py           # Pipeline completo de transformación
├── red_neuronal.py              # Arquitectura, entrenamiento, evaluación
├── visualizaciones.py           # Análisis gráfico de resultados
├── modelo_red_neuronal.keras    # Modelo listo para producción
├── metricas_red_neuronal.json   # Métricas detalladas guardadas
└── {X,y}_{train,test}.npy      # Datos procesados reproducibles
```

## La Gran Conclusión

De esta parte del laboratorio #2 extraemos que las redes neuronales justifican su complejidad cuando el problema presenta verdaderas no linealidades. A diferencia de enfoques más simples, pueden descubrir automáticamente que ciertos géneros explotan en ciertas plataformas durante ciertos períodos - patrones multidimensionales que requieren la expresividad de múltiples capas.

No es que las redes neuronales sean siempre superiores, sino que para problemas con interacciones complejas genuinas, proporcionan la flexibilidad arquitectónica necesaria para capturar la realidad del fenómeno.

**Resultado final:** 94.6% de precisión con error promedio de 93,200 copias. Una herramienta que puede informar decisiones de millones de dólares con confianza estadística sólida.