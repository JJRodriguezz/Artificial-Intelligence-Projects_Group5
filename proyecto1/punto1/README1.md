# Algoritmo A* - Búsqueda de Ruta Óptima en Romania

Este proyecto implementa y compara dos algoritmos de búsqueda para encontrar la ruta más corta entre ciudades dadas por el problema: **Best-First Search** y **A* Search**.

##  Análisis del Problema

El problema consiste en encontrar la ruta de menor costo entre dos ciudades en un grafo ponderado no dirigido que representa un mapa con diferentes ciudades interconectadas.

### Características del problema:
- **Nodos**: Ciudades (Arad, Sibiu, Bucharest, etc.)
- **Aristas**: Conexiones entre ciudades con distancias en kilómetros (a lo largo del programa tomamos dicha unidad de medida como referencia)
- **Objetivo**: Encontrar el camino más corto desde Arad hasta Bucharest
- **Tipo**: Problema de búsqueda en espacio de estados con costo

```python
# Definición del problema
initial = 'Arad'
goal = 'Bucharest'

# Ejemplo de conexiones
actions = {
    'Arad': ['Sibiu', 'Timisoara', 'Zerind'],
    'Sibiu': ['Arad', 'Fagaras', 'Rimnicu Vilcea', 'Oradea'],
    # ...
}

# Ejemplo de costos
action_costs = {
    ('Arad', 'Sibiu'): 140,
    ('Arad', 'Timisoara'): 118,
    # ...
}
```

## Cómo se Aplica A*

### Función de Evaluación
A* utiliza la función: **f(n) = g(n) + h(n)**

```python
def f_astar(node):
    """Función de evaluación para A*: f(n) = g(n) + h(n)"""
    return node.path_cost + h(node)  # g(n) + h(n)
```

Donde:
- **g(n)**: Costo real acumulado desde Arad hasta el nodo actual
- **h(n)**: Heurística (distancia en línea recta hasta Bucharest)
- **f(n)**: Estimación del costo total del camino más barato

### Heurística Utilizada
Se usa la distancia euclidiana (línea recta) desde cada ciudad hasta Bucharest:

```python
# Heurística: distancia en línea recta a Bucharest
heuristic = {
    'Arad': 366,
    'Sibiu': 253,
    'Pitesti': 100,
    'Bucharest': 0,
    # ...
}

def h(node):
    """Función heurística: distancia en línea recta hasta Bucharest"""
    return heuristic.get(node.state, 0)
```

### Proceso de Búsqueda
```python
def a_star_search(problem, h_func):
    node = Node(state=problem.initial)
    frontier = [(f_astar(node), node)]  # Cola de prioridad ordenada por f(n)
    
    while frontier:
        _, node = heapq.heappop(frontier)
        
        if problem.is_goal(node.state):
            return node
            
        for child in expand(problem, node):
            # Evaluar cada hijo con f(n) = g(n) + h(n)
            f_value = child.path_cost + h_func(child)
            heapq.heappush(frontier, (f_value, child))
```

## ¿Por qué la Ruta Encontrada es Óptima?

### 1. Garantía Teórica
A* garantiza encontrar la solución óptima cuando la heurística es:
- **Admisible**: Nunca sobreestima el costo real
- **Consistente**: Cumple la desigualdad triangular

### 2. Heurística Admisible
La distancia en línea recta nunca puede ser mayor que la distancia real por carretera:

```
Ejemplo:
- Línea recta Arad → Bucharest: 366 km
- Ruta real más corta: 418 km (siempre ≥ 366 km)
```

### 3. Exploración Sistemática
A* explora los nodos en orden de f(n) creciente, garantizando que cuando encuentra el objetivo, no existe un camino mejor sin explorar.

### 4. Verificación Práctica
La ruta encontrada: **Arad → Sibiu → Rimnicu Vilcea → Pitesti → Bucharest (418 km)**

Comparada con alternativas:
- Arad → Timisoara → ... → Bucharest: 673 km
- Arad → Zerind → ... → Bucharest: 646 km

## Ejecución del Código

### Resultado Esperado
```
--- BEST-FIRST SEARCH ---
Camino: Arad -> Sibiu -> Rimnicu Vilcea -> Pitesti -> Bucharest
Costo total: 418
Nodos explorados: 6

--- A* SEARCH ---
Camino: Arad -> Sibiu -> Rimnicu Vilcea -> Pitesti -> Bucharest
Costo total: 418
Nodos explorados: 4

--- COMPARACION ---
A* fue mas eficiente por 2 nodos
Ambos encontraron el mismo costo optimo: 418
```

### Análisis de Eficiencia
- **A***: Más eficiente, explora menos nodos irrelevantes
- **Best-First**: Explora más exhaustivamente sin guía heurística
- **Resultado**: Ambos encuentran la solución óptima, pero A* es más inteligente

## Conclusión

A* demuestra ser superior a Best-First Search en eficiencia de exploración, utilizando la información heurística para dirigirse más directamente hacia el objetivo, garantizando siempre la solución óptima cuando se usa una heurística admisible y consistente.