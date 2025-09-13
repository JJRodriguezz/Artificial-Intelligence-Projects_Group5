# Análisis de Ventas de Videojuegos con SVM
##  ¿Qué estamos haciendo aquí?

A partir de los datos de las ventas de los últimos años de los que dispone este dataset que elegimos, queremos predecir cuánto venderá globalmente un juego nuevo. En este apartado del laboratorio #2 abordaremos cómo dimos solución a esta idea/pregunta haciendo uso de un modelo de aprendizaje supervisado de tipo SVM...

### El Dataset: 
La fuente de todo este informe y análisis. Trabajamos con el dataset vgsales.csv de Kaggle, que contiene:

16,598 juegos analizados

11 características por juego:

Rank, Name, Platform, Year, Genre, Publisher

NA_Sales (ventas en Norteamérica)

EU_Sales (ventas en Europa)

JP_Sales (ventas en Japón)

Other_Sales (ventas en otras regiones)

Global_Sales (nuestra variable objetivo)

#### Dato curioso que encontramos: 
Las ventas promedio son de 0.54 millones por juego, pero hay desde pequeños indie que venden 0.01 millones hasta gigantes como Wii Sports que alcanzan 82.74 millones.

## El Algoritmo: Por qué elegimos SVM
Seleccionamos para esta parte Support Vector Machine porque sabemos que es particularmente bueno para:

Problemas de regresión como este

Datos con relaciones complejas entre características

Evitar overfitting cuando se configura adecuadamente

Pero aquí viene la parte interesante...

### ¿Cómo llegamos a una respuesta optima?: Kernel Wars
Probamos varios enfoques y aquí presentamos un pequeño resumen de lo que encontramos:

- El Ganador: Kernel Linear
R² Score: 0.9986 (prácticamente perfecto)

- RMSE: 0.0767 (error muy bajo)

- MAE: 0.0759 (nuestras predicciones están muy cerca de la realidad)

El kernel linear demostró ser increíblemente efectivo. ¿Por qué? Porque las relaciones entre las ventas regionales y las globales son esencialmente lineales (las globales son la suma de las regionales). A veces la solución más simple es la mejor.

### Los Perdedor (descartado): Poly

#### Kernel Polynomial:

- R² de -48.6 (resultado negativo) lo cual concluimos que era un resultado desastroso y decidimos descartarlo.

- Tiempo de entrenamiento extremadamente largo para las predicciones obtenidas.

- Completamente inútil para este problema, ya que no hallaba predicciones acertadas, de hecho eran muy lejanas a la realidad.

### ¿Por qué fracasó? 
Este kernel intentaba encontrar patrones complejos donde no los había, sobrecomplicando el problema y cayendo en overfitting (que como ya sabemos, es contrario a lo que algunos creerían, lejos de generar una buena base de predicción termina siendo contraproducente para los análisis y predicciones según lo estudiado en el curso).

## Entendiendo los Resultados
### El Bueno:
- R² de 0.9986 significa que nuestro modelo explica el 99.86% de la variabilidad en las ventas globales

- Errores absolutos pequeños (0.07-0.09 millones en promedio)

#### Explicación para algunos resultados de error muy alto:

Algunos errores porcentuales parecen altos (200-300%), pero esto resulta siendo un poco engañoso para nosotros. 
Lo que pasa es que esto sucede con juegos que vendieron muy poco (0.02-0.03 millones), donde un error de 0.07 millones se convierte en un porcentaje grande. En términos absolutos, el error es mínimo.

## ¿Qué podemos concluir?
- Las ventas regionales son predictores excelentes de las globales (correlaciones de 0.94, 0.90, 0.61, 0.74)

#### "Keep it simple, stupid" 
- El kernel linear superó por mucho a opciones más complejas

- SVM es efectivo para problemas de regresión con relaciones lineales claras

- El tiempo de entrenamiento importa: No vale la pena esperar horas por resultados no tan buenos cuando no estamos implementando el algoritmo correcto para el problema que estamos abordando.

### Aplicaciones Prácticas
Este modelo podría ayudar a:

- Editoras a predecir el potencial de nuevos juegos.

- Desarrolladores a entender qué mercados priorizar.

- Analistas a identificar patrones de ventas por región.

De esta parte del laboratorio #2 podemos extraer que no siempre es necesario utilizar el modelo o algortimo más complicado para todas las tareas, con la implementación de este SVM desde diferentes perspectivas (kernels) pudimos evidenciar que a veces soluciones más simples llevan a resultados acertados. Seguramente habrán modelos más complejos que se acercarán más a la perfección que el que seleccionamos con ganador en esta parte, pero no está nada mal para una solución y respuesta rápida.