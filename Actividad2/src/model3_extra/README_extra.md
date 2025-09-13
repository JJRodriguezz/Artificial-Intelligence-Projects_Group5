# Análisis de Ventas de Videojuegos con Gradient Boosting
##  ¿Qué estamos haciendo aquí?

Queremos predecir las **ventas globales de videojuegos** usando datos históricos del dataset *vgsales.csv*
Tomado de: https://www.kaggle.com/datasets/gregorut/videogamesales  
Para este tercer punto usamos el modelo **Gradient Boosting Regressor**, un algoritmo de aprendizaje supervisado que combina varios árboles de decisión para generar predicciones más precisas.

---

### El Dataset
La base de este análisis es el dataset **vgsales_limpio.csv**, con:

- **16,598 juegos** analizados  
- **11 características** por juego:  
  Rank, Name, Platform, Year, Genre, Publisher  
  NA_Sales, EU_Sales, JP_Sales, Other_Sales  
  Global_Sales (variable objetivo)  

#### Dato curioso que encontramos:
Las ventas promedio son de 0.54 millones por juego, pero hay desde pequeños indie que venden 0.01 millones hasta gigantes como Wii Sports que alcanzan 82.74 millones.

---

## El Algoritmo: Por qué elegimos Gradient Boosting
Elegimos **Gradient Boosting** porque luego de tomar decision sobre otros algoritmos como el XGBoost o el catBoost el **Gradiente Boosting Regressor:**

- Combina **muchos modelos débiles** (árboles) para crear uno fuerte  
- Permite controlar el **sobreajuste** mediante hiperparámetros (learning_rate, n_estimators, max_depth)  
- Se adapta bien a datos con **relaciones no lineales**  

Además, es ampliamente utilizado en la práctica y competiciones como Kaggle, destacando por su **precisión y flexibilidad**.

---

### Resultados principales
Entrenamos el modelo con **n_estimators=400**, **learning_rate=0.05** y **max_depth=3**.  
En el desarrollo y entorno de prueba obtuvimos los siguientes resultados:

- **R² Score**: 0.8638  
- **RMSE**: 0.7565  
- **MAE**: 0.0178  

Estos valores indican:
- R² cercano a 1 Lo que nos representa que el modelo explica un **86%** de la variabilidad en las ventas globales.  
- RMSE y MAE bajos Lo que nos representa errores pequeños en promedio, nuestras predicciones están muy cerca de la realidad.  

---

### Visualizaciones
Incluimos las siguientes gráficas para entender mejor el modelo:

1. **Predicciones vs Valores Reales**:  
   Muestra qué tan cerca están las predicciones de los valores reales.  

2. **Distribución de Residuos**:  
   Los errores se concentran cerca de cero, es una señal de buen ajuste.  

3. **Importancia de Features**:  
   `Rank` y `NA_Sales` son las variables más influyentes.  

4. **Residuos vs Predicciones**:  
   Verifica que los errores no sigan patrones extraños.  

![Análisis Gradient Boosting](reports/gboost_analysis.png)

---

## Interpretación y Conclusiones

* Las **ventas por región** y el **ranking** son lo que más pesa para explicar las ventas globales.
* El modelo toma bien las relaciones importantes como que es preciso y no se enreda tanto con el sobreajuste.
* Los errores grandes pasan sobre todo con juegos que no se venden mucho, porque un pequeño error en porcentaje se nota mucho

---

### ¿Para qué sirve esto?

El modelo nos puede ser útil para:

* **Editoras**: darse una idea de cuánto pueden vender antes de sacar un juego.
* **Desarrolladores**: decidir en qué mercados enfocarse.
* **Analistas**: analizar patrones de ventas según la región.

En resumen, **Gradient Boosting** es una buena opción:
Es fácil de programar, rápido para entrenarlo y es bastante preciso a la hora de hacer las predicciones.

Además concluimos que no hace falta hacer un modelo super complicado o super extenso para analizar de forma correcta y tener buenos resultados