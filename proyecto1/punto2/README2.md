
# Ejercicio 2
## Problema del Maze

El problema de Maze, establece el problema de un robot que busca salir de un laberinto con una representación de un laberinto en 2D.

## Análisis y Respuestas

### 1. Resolver el ejercicio planteado

El algoritmo utilizado para resolver este problema hace uso de una heurística llamada Distancia manhattan que utiliza las coordenadas actuales del robot y las coordenadas de la salida, con el fin de establece un valor numérico correspondiente a la distancia usando líneas horizontales y verticales entre ambos puntos.

En este ejercicio cada casilla me representa un estado del problema, estado que se representa como una coordenada **(fila, columna)** que me indica la posición actual del robot dentro del laberinto. Desde cualquier estado el robot tiene la posibilidad de moverse hacia arriba, abajo, izquierda o derecha; Siempre y cuando las casillas vecinas o de destinos estén libres y no sean paredes, además, los movimientos en diagonal no están permitidos.
La función de transición llamada **get_neighbors()** es la encargada de calcular a qué posiciones puedo llegar desde el estado actual y me descarta aquellas a las que no puedo ir.
El costo es uniforme, lo que significa que cada movimiento tiene un costo de 1 ya que el algoritmo busca minimizar el número de pasos hasta el final.
La clase **Problem:** es la que me encapsula la definición del laberinto: sus dimensiones, las acciones permitidas definidas anteriormente y puntos clave como el inicio y el final.
Especificamente el código para resolver este problema lo podrá encontrar en el archivo denominado **Maze_problem.py** en el actual repositorio.

### 2. ¿Cómo cambia el comportamiento del algoritmo si cambiamos la función de costo?

El comportamiento del algoritmo depende directamente de la función de costo utilizada. Si todas las acciones tienen el mismo costo (por ejemplo, moverse a cualquier casilla cuesta 1), el algoritmo buscará el camino más corto en número de pasos. Si se asignan diferentes costos a ciertas acciones o casillas (por ejemplo, moverse sobre otro obstaculo cuesta 2 y sobre camino cuesta 1), el algoritmo tenderá a evitar los caminos más costosos, aunque sean más cortos en distancia, priorizando el menor costo total. Esto puede cambiar la ruta óptima encontrada y el tiempo de búsqueda.

### 3. ¿Qué sucede si hay múltiples salidas en el laberinto? ¿Cómo podrías modificar el algoritmo para manejar esto? Plantea una propuesta.

Si hay múltiples salidas, el algoritmo implementado solo encuentra la ruta hacia una salida específica que es la que se define como **end**. Para manejar múltiples salidas, se puede modificar el algoritmo para considerar como final cualquier casilla que sea una salida. Una propuesta sería implementar un algoritmo tradicional que calcule la salida más cercana de la posición actual del robot y luego calcule la ruta hacia esta. Esto permite que el algoritmo encuentre la salida más cercana o la de menor costo, dependiendo de la heurística y función de costo.

### 4. Modifica el laberinto por uno más grande y con otro tipo de obstáculo además de paredes. ¿Qué limitación encuentras en el algoritmo?

Al modificar el laberinto para que sea más grande y agregar obstáculos adicionales, por ejemplo, zonas de pasto, zonas de agua, o simplemente trampas que tengan un costo mayor o sean intransitables, el algoritmo puede enfrentar las siguientes limitaciones:

- **Eficiencia**: En laberintos muy grandes, la cantidad de nodos que tendriamos que explorar crecería rápidamente, lo que podría hacer que el algoritmo sea lento o consuma mucha memoria.
- **Flexibilidad**: Si el algoritmo solo reconoce paredes como obstáculos, habría que modificar la función **passable()** y la función de costo para manejar nuevos tipos de obstáculos.
- **Representación**: Es necesario adaptar la representación del laberinto para distinguir entre diferentes tipos de casillas y sus costos. Por ejemplo definiendo el "_" como un nuevo obstaculo, transitable pero con mayor costo

#### Conclusión:
El algoritmo implementado, que es el algoritmo A* es flexible y eficiente para laberintos de tamaño moderado y con obstáculos simples, cómo el presentado en el ejemplo de uso y en el problema presentado. Pero tal cuál como está requeriría agregar mejoras y más implementaciones para laberintos más grandes y complejos.