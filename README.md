# Proyecto de Inteligencia Artificial - Algoritmos de Búsqueda

## Descripción General

Este repositorio contiene la implementación y análisis de tres ejercicios fundamentales sobre algoritmos de búsqueda en inteligencia artificial. Los ejercicios abordan diferentes problemas de pathfinding y navegación utilizando algoritmos clásicos como A*, BFS e IDS.

## Estructura del Proyecto

```
├── ejercicio1/
│   ├── 2.BestFirstSearch.ipynb
│   └── resultados/
├── ejercicio2/
│   ├── Maze_problem.ipynb
│   └── mapas/
├── ejercicio3/
│   ├── metro_navigation.ipynb
│   └── datos/
└── README.md
```

## Ejercicios

### Ejercicio 1: Algoritmo A* - Ruta a Bucharest

**Objetivo:** Implementar el algoritmo A*Search para encontrar la ruta óptima hasta Bucharest utilizando heurísticas.

**Descripción:** 
- Parte del notebook base "2.BestFirstSearch.ipynb" de la semana 2
- Implementa el algoritmo A* con función heurística
- Encuentra la ruta óptima considerando tanto el costo real como la estimación heurística
- Compara resultados con otros algoritmos de búsqueda

**Archivos:**
- `2.BestFirstSearch.ipynb` - Implementación del algoritmo A*
- `resultados/` - Visualizaciones y análisis de rutas

### Ejercicio 2: Navegación en Laberinto

**Objetivo:** Resolver problemas de navegación en laberintos y analizar el comportamiento del algoritmo bajo diferentes condiciones.

**Componentes del ejercicio:**
1. **Resolución básica:** Implementar algoritmo de búsqueda para encontrar salida del laberinto
2. **Análisis de función de costo:** Estudiar cómo diferentes funciones de costo afectan el comportamiento
3. **Múltiples salidas:** Proponer modificaciones para manejar laberintos con varias salidas
4. **Laberinto extendido:** Crear laberintos más grandes con obstáculos adicionales y analizar limitaciones

**Preguntas de investigación:**
- ¿Cómo cambia el comportamiento del algoritmo si cambiamos la función de costo?
- ¿Qué sucede si hay múltiples salidas en el laberinto?
- ¿Qué limitaciones encuentra el algoritmo en laberintos complejos?

**Archivos:**
- `Maze_problem.ipynb` - Implementación y análisis completo
- `mapas/` - Diferentes configuraciones de laberintos

### Ejercicio 3: Sistema de Navegación de Metro

**Contexto:** Desarrollo de un sistema de navegación para la red de metro de una ciudad importante.

**Objetivo:** Implementar algoritmos BFS e IDS para encontrar la ruta más corta entre estaciones de metro.

**Características:**
- Utiliza **Breadth-First Search (BFS)** para búsqueda óptima
- Implementa **Iterative Deepening Search (IDS)** como alternativa
- Encuentra rutas con menor número de estaciones (acciones)
- Compara eficiencia y resultados de ambos algoritmos

**Entregables:**
- Algoritmo BFS para navegación en red de metro
- Algoritmo IDS para el mismo problema
- Análisis comparativo de ambos enfoques
- Visualización de rutas encontradas

**Archivos:**
- `metro_navigation.ipynb` - Implementación completa del sistema
- `datos/` - Mapas y datos de la red de metro

## Tecnologías Utilizadas

- **Python 3.x**
- **Jupyter Notebooks**
- **NumPy** - Operaciones numéricas
- **Matplotlib** - Visualizaciones
- **NetworkX** - Manejo de grafos (si aplica)
- **Pandas** - Manipulación de datos

## Algoritmos Implementados

1. **A* (A-Star)** - Búsqueda informada con heurística
2. **BFS (Breadth-First Search)** - Búsqueda en anchura
3. **IDS (Iterative Deepening Search)** - Búsqueda con profundización iterativa
4. **Best-First Search** - Búsqueda voraz informada

## Ejecución

Para ejecutar cualquier ejercicio:

1. Instalar dependencias:
```bash
pip install jupyter numpy matplotlib pandas networkx
```

2. Abrir Jupyter Notebook:
```bash
jupyter notebook
```

3. Navegar al ejercicio deseado y ejecutar las celdas correspondientes.

## Resultados y Análisis

Cada ejercicio incluye:
- Implementación completa del algoritmo
- Análisis de complejidad temporal y espacial
- Visualizaciones de resultados
- Comparaciones entre diferentes enfoques
- Discusión de limitaciones y posibles mejoras

## Autores
- Isabella Idarraga
- Juan Jose Rodriguez
- Diego 
Desarrollado como parte del curso de Inteligencia Artificial - [2025]

