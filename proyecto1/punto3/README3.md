# Sistema de Navegación en Red de Metro

## Contexto del Problema:

Como desarrollador de la alcaldía de una importante ciudad, se le solicita implementar un algoritmo que permita a los pasajeros encontrar la ruta más corta entre dos estaciones del
metro usando dos estrategias diferentes **(Breadth-First Search (BFS) e Iterative Deepening Search (IDS))**. Se le suministrará un mapa de la red de metro, y debe determinar la ruta con menos acciones (estaciones de parada) entre dos estaciones usando BFS e IDS.


## Mapa de la Red de Metro

La red de metro cuenta con **10 estaciones** conectadas de la siguiente forma:

```
     A
   /   \
  B     C
 /|\     \
D E      F
| |\      \
G H I ---- J
    
```

**Conexiones detalladas:**
- **Estación A**: B, C
- **Estación B**: A, D, E
- **Estación C**: A, F
- **Estación D**: B, G
- **Estación E**: B, H, I
- **Estación F**: C, J
- **Estación G**: D
- **Estación H**: E
- **Estación I**: E, J
- **Estación J**: F, I

## Definición del Problema

### Estados y Componentes
**1. Estado Inicial:** La estación donde comienza el pasajero.

**2. Estado Objetivo:** La estación a la que el pasajero quiere llegar.

**3. Acciones:** Desde cada estación, el pasajero puede moverse a
cualquier estación conectada directamente.

**4. Espacio de Estados:** Todas las posibles combinaciones de estaciones
y movimientos entre ellas.

**5. Modelo de Transición:** El estado resultante después de moverse de
una estación a otra.

**6. Costo**: Uniforme entre todas las estaciones (1 por cada movimiento)

## Clases en el código

### Clase `Node`
Representa un nodo en el árbol de búsqueda:
```python
class Node:
    def __init__(self, state, path, depth, parent):
        self.state = state      # Estación actual
        self.path = path        # Camino recorrido
        self.depth = depth      # Profundidad en el árbol
        self.parent = parent    # Nodo padre
```

### Clase `MetroProblem`
Define el problema y las operaciones básicas:
```python
class MetroProblem:
    def actions(self, state)           # Estaciones conectadas
    def result(self, state, action)    # Nueva estación después del movimiento
    def goal_test(self, state, goal)   # Verificar si se alcanzó el objetivo
```

### Clase `SearchAlgorithms`
Implementa los algoritmos de búsqueda BFS e IDS.

## Algoritmos Implementados

### 1. Breadth-First Search (BFS)

**Características:**
- Explora todos los nodos de un nivel antes de pasar al siguiente
- Cola (FIFO - First In, First Out)
- Garantiza encontrar la ruta más corta
- Siempre encuentra una solución si existe

**Funcionamiento:**
1. Inicia con el nodo raíz en la cola
2. Extrae el primer nodo de la cola
3. Si es el objetivo, termina
4. Agrega todos los hijos no visitados a la cola
5. Repite hasta encontrar la solución

**Complejidad:**
- **Tiempo**: O(b^d) 
- **Espacio**: O(b^d) 

### 2. Iterative Deepening Search (IDS)

**Características:**
- Combina DFS con exploración sistemática por niveles
- Ejecuta DFS con límites de profundidad crecientes
- Garantiza encontrar la ruta más corta
- Utiliza menos memoria que BFS

**Funcionamiento:**
1. Ejecuta DFS con límite de profundidad 0, 1, 2, ...
2. En cada iteración, explora hasta la profundidad límite
3. Si no encuentra solución, incrementa el límite
4. Repite hasta encontrar la solución

**Complejidad:**
- **Tiempo**: O(b^d)
- **Espacio**: O(bd)

## Resultados y Análisis

### Caso de Prueba Principal: Estación A → Estación J

**Ruta Óptima Encontrada:**
```
A → C → F → J
```
- **Número de paradas**: 3
- **Estaciones visitadas**: 4 (incluyendo origen y destino)

### Comparación de Rendimiento en mi dispositivo

| Métrica | BFS | IDS | Ganador |
|---------|-----|-----|---------|
| **Tiempo de Ejecución** | 0.000734s | 0.000989s | BFS |
| **Memoria Utilizada** | 5.69 KB | 1.07 KB | IDS |
| **Nodos Explorados** | 7 | 20 | BFS |
| **Tamaño Máx. Frontera** | 4 | 0 | IDS |
| **Iteraciones de Profundidad** | - | 4 | - |
| **Optimalidad** | Sí | Sí | Empate |

### Análisis de Diferentes Rutas

| Ruta | BFS (Nodos) | IDS (Nodos) | Diferencia |
|------|-------------|-------------|------------|
| A → B | 2 | 3 | +50% |
| A → J | 7 | 20 | +186% |
| G → H | 6 | 18 | +200% |
| C → I | 6 | 17 | +183% |

### Patrón de Exploración Observado

**BFS - Exploración Sistemática por Niveles:**
- Profundidad 0: A
- Profundidad 1: B, C
- Profundidad 2: D, E, F
- Encuentra J en profundidad 3

**IDS - Exploración Iterativa:**
- Iteración 1 (profundidad 0): 1 nodo explorado
- Iteración 2 (profundidad 1): 3 nodos explorados total
- Iteración 3 (profundidad 2): 6 nodos explorados total
- Iteración 4 (profundidad 3): Encuentra solución con 20 nodos explorados total

## Conclusiones

### Ventajas del BFS
- Explora menos nodos 
- No repite exploraciones
- Comportamiento consistente
- Siempre encuentra la ruta más corta

### Ventajas del IDS
- **Menor uso de memoria** - O(bd) vs O(b^d)
- Mejor para grafos muy profundos
- Puede interrumpirse en cualquier momento


### Diferencias Clave (Porcentajes calculados según resultados en mi dispositivo)

1. **Exploración de Nodos**: BFS explora significativamente menos nodos que IDS. IDS explora entre 50% y 200% más nodos porque re-explora nodos en cada iteración de profundidad menor.

2. **Uso de Memoria**: IDS utiliza aproximadamente 81% menos memoria que BFS (1.07 KB vs 5.69 KB) porque mantiene el camino actual, no toda la frontera.

3. **Tiempo de Ejecución**: BFS es aproximadamente 26% más rápido (0.000734s vs 0.000989s) porque no tiene que reiniciar la búsqueda en cada iteración.

4. **Iteraciones de Profundidad**: IDS hizo 4 iteraciones para encontrar la solución A → J, exploró profundidades 0, 1, 2, y despues encontró la solución en profundidad 3.

5. **Aplicabilidad Práctica**: 
   - **BFS es mejor** para grafos pequeños o medianos donde la memoria no este limitada y se requiera resolver en menor tiempo
   - **IDS es mejor** para grafos grandes o cuando la memoria es muy limitada y permita mayor exploración de nodos


### Funcionalidades
1. **Comparación principal**: Ejecuta BFS e IDS para la ruta A → J
2. **Análisis de múltiples rutas**: Prueba diferentes combinaciones de origen-destino
3. **Métricas de rendimiento**: Tiempo, memoria, nodos explorados
4. **Visualización de resultados**: Salida con análisis comparativo

