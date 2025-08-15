import heapq  # El módulo heapq para implementar colas de prioridad (heaps)

class Node:  # definición de clase node
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state  # El estado que define el nodo
        self.parent = parent  # El nodo padre de donde se origina el nodo actual
        self.action = action  # Action tomada desde el padre para llegar al nodo
        self.path_cost = path_cost  # costo desde el nodo raíz (estado inicial), hasta el nodo actual

    def __lt__(self, other):  # comparar dos objetos de clase node basado en el costo
        return self.path_cost < other.path_cost

def expand(problem, node):
    s = node.state
    for action in problem.actions(s):
        s_prime = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s_prime)
        yield Node(state=s_prime, parent=node, action=action, path_cost=cost)

class Problem:  # DEFINICION DEL PROBLEMA
    def __init__(self, initial, goal, actions, result, action_cost, is_goal):
        self.initial = initial  # Estado inicial
        self.goal = goal  # Estado objetivo
        self.actions = actions  # acciones disponibles desde un estado.
        self.result = result  # estado resultante de aplicar una acción
        self.action_cost = action_cost  # costo de una acción
        self.is_goal = is_goal  # verificación de si el estado es el estado objetivo

def best_first_search(problem, f):
    node = Node(state=problem.initial)  # Crea el nodo raíz con el estado inicial del problema.
    frontier = [(f(node), node)]  # frontera como una cola de prioridad (f(n)) con el nodo inicial.
    heapq.heapify(frontier)  # Convierte la lista frontier en una cola de prioridad (heap)
    reached = {problem.initial: node}  # registrar los estados alcanzados y su nodo correspondiente.
    
    # CONTADORES Y TRACKING PARA ANÁLISIS (No requerido pero lo agregamos como adicional)
    nodes_explored = 0
    exploration_order = []
    frontier_history = []

    while frontier:
        # Registrar estado actual de la frontera
        frontier_states = [node.state for _, node in frontier[:5]]  # Primeros 5 para no saturar
        frontier_history.append(frontier_states.copy())
        
        _, node = heapq.heappop(frontier)  # Extrae el nodo con el valor mínimo de f de la frontera.
        nodes_explored += 1
        exploration_order.append({
            'state': node.state,
            'path_cost': node.path_cost,
            'f_value': f(node),
            'step': nodes_explored
        })
        
        if problem.is_goal(node.state):   # Si el estado del nodo es el estado objetivo, devuelve el nodo.
            # Agregar estadísticas al nodo solución
            node.nodes_explored = nodes_explored
            node.exploration_order = exploration_order
            node.frontier_history = frontier_history
            return node

        for child in expand(problem, node):  # Expande el nodo generando sus nodos hijos.
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:  # Si el estado del nodo hijo no ha sido alcanzado antes o si se alcanza con un costo de camino menor, actualiza el dict y añade el nodo hijo a la frontera.
                reached[s] = child
                heapq.heappush(frontier, (f(child), child))  # Añade el nodo hijo a la frontera

    return None  # Se exploran todos los nodos posibles, y no se encuentra una solución

# Definición de funciones específicas del problema
def result(state, action):
    return action

def action_cost(state, action, result_state):
    return action_costs.get((state, action), float('inf'))  # En el caso de que no se encuentre un costo, el valor sera infinito

def is_goal(state):
    return state == goal

def f(node):
    return node.path_cost  # costo del camino desde el estado inicial hasta el nodo actual.

# Definición del problema de Romania
initial = 'Arad'
goal = 'Bucharest'

actions = {
    # Completar con las acciones disponibles desde cada estado
    'Arad': ['Sibiu', 'Timisoara', 'Zerind'],
    'Sibiu': ['Arad', 'Fagaras', 'Rimnicu Vilcea', 'Oradea'],
    'Timisoara': ['Arad', 'Lugoj'],
    'Zerind': ['Arad', 'Oradea'],
    'Fagaras': ['Sibiu', 'Bucharest'],
    'Rimnicu Vilcea': ['Sibiu', 'Pitesti', 'Craiova'],
    'Lugoj': ['Timisoara', 'Mehadia'],
    'Oradea': ['Zerind', 'Sibiu'],
    'Pitesti': ['Rimnicu Vilcea', 'Bucharest', 'Craiova'],
    'Craiova': ['Rimnicu Vilcea', 'Drobeta', 'Pitesti'],
    'Mehadia': ['Lugoj', 'Drobeta'],
    'Drobeta': ['Mehadia', 'Craiova'],
    'Bucharest': ['Fagaras', 'Pitesti', 'Urziceni', 'Giurgiu'],
    'Urziceni': ['Bucharest', 'Hirsova', 'Vaslui'],
    'Giurgiu': ['Bucharest'],
    'Hirsova': ['Urziceni', 'Eforie'],
    'Eforie': ['Hirsova'],
    'Vaslui': ['Urziceni', 'Iasi'],
    'Iasi': ['Vaslui', 'Neamt'],
    'Neamt': ['Iasi']
}

action_costs = {
    # Costos bidireccionales
    ('Arad', 'Sibiu'): 140, ('Sibiu', 'Arad'): 140,
    ('Arad', 'Timisoara'): 118, ('Timisoara', 'Arad'): 118,
    ('Arad', 'Zerind'): 75, ('Zerind', 'Arad'): 75,
    ('Sibiu', 'Fagaras'): 99, ('Fagaras', 'Sibiu'): 99,
    ('Sibiu', 'Rimnicu Vilcea'): 80, ('Rimnicu Vilcea', 'Sibiu'): 80,
    ('Sibiu', 'Oradea'): 151, ('Oradea', 'Sibiu'): 151,
    ('Timisoara', 'Lugoj'): 111, ('Lugoj', 'Timisoara'): 111,
    ('Zerind', 'Oradea'): 71, ('Oradea', 'Zerind'): 71,
    ('Fagaras', 'Bucharest'): 211, ('Bucharest', 'Fagaras'): 211,
    ('Rimnicu Vilcea', 'Pitesti'): 97, ('Pitesti', 'Rimnicu Vilcea'): 97,
    ('Rimnicu Vilcea', 'Craiova'): 146, ('Craiova', 'Rimnicu Vilcea'): 146,
    ('Lugoj', 'Mehadia'): 70, ('Mehadia', 'Lugoj'): 70,
    ('Pitesti', 'Bucharest'): 101, ('Bucharest', 'Pitesti'): 101,
    ('Pitesti', 'Craiova'): 138, ('Craiova', 'Pitesti'): 138,
    ('Craiova', 'Drobeta'): 120, ('Drobeta', 'Craiova'): 120,
    ('Mehadia', 'Drobeta'): 75, ('Drobeta', 'Mehadia'): 75,
    ('Bucharest', 'Urziceni'): 85, ('Urziceni', 'Bucharest'): 85,
    ('Bucharest', 'Giurgiu'): 90, ('Giurgiu', 'Bucharest'): 90,
    ('Urziceni', 'Hirsova'): 98, ('Hirsova', 'Urziceni'): 98,
    ('Urziceni', 'Vaslui'): 142, ('Vaslui', 'Urziceni'): 142,
    ('Hirsova', 'Eforie'): 86, ('Eforie', 'Hirsova'): 86,
    ('Vaslui', 'Iasi'): 92, ('Iasi', 'Vaslui'): 92,
    ('Iasi', 'Neamt'): 87, ('Neamt', 'Iasi'): 87
}

# Heurística: distancia en línea recta a Bucharest (admisible y consistente)
heuristic = {
    'Arad': 366,
    'Bucharest': 0,
    'Craiova': 160,
    'Drobeta': 242,
    'Eforie': 161,
    'Fagaras': 176,
    'Giurgiu': 77,
    'Hirsova': 151,
    'Iasi': 226,
    'Lugoj': 244,
    'Mehadia': 241,
    'Neamt': 234,
    'Oradea': 380,
    'Pitesti': 100,
    'Rimnicu Vilcea': 193,
    'Sibiu': 253,
    'Timisoara': 329,
    'Urziceni': 80,
    'Vaslui': 199,
    'Zerind': 374
}

def h(node):
    """Función heurística: distancia en línea recta hasta Bucharest"""
    return heuristic.get(node.state, 0)

def f_astar(node):
    """Función de evaluación para A*: f(n) = g(n) + h(n)"""
    return node.path_cost + h(node)  # g(n) + h(n)

def a_star_search(problem, h_func):

    node = Node(state=problem.initial)
    frontier = [(f_astar(node), node)]  # Cola de prioridad ordenada por f(n) = g(n) + h(n)
    heapq.heapify(frontier)
    reached = {problem.initial: node}
    
    # CONTADORES Y TRACKING PARA ANÁLISIS
    nodes_explored = 0
    exploration_order = []
    frontier_history = []
    
    while frontier:
        # Registrar estado actual de la frontera
        frontier_states = [node.state for _, node in frontier[:5]]  # Primeros 5 para no saturar
        frontier_history.append(frontier_states.copy())
        
        _, node = heapq.heappop(frontier)
        nodes_explored += 1
        h_val = h_func(node)
        f_val = node.path_cost + h_val
        
        exploration_order.append({
            'state': node.state,
            'path_cost': node.path_cost,
            'heuristic': h_val,
            'f_value': f_val,
            'step': nodes_explored
        })
        
        if problem.is_goal(node.state):
            # Agregar estadísticas al nodo solución
            node.nodes_explored = nodes_explored
            node.exploration_order = exploration_order
            node.frontier_history = frontier_history
            return node
            
        for child in expand(problem, node):
            s = child.state
            # Si no hemos alcanzado este estado o lo alcanzamos con menor costo
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                f_value = child.path_cost + h_func(child)  # f(n) = g(n) + h(n)
                heapq.heappush(frontier, (f_value, child))
    
    return None

# ===========================================
# EJECUCIÓN DE AMBOS ALGORITMOS
# ===========================================

# Crear el problema
problem = Problem(
    initial, 
    goal, 
    lambda s: actions.get(s, []), 
    result, 
    action_cost, 
    is_goal
)

print("=" * 60)
print("COMPARACIÓN: BEST-FIRST SEARCH vs A* SEARCH")
print("=" * 60)

# ===============================
# BEST-FIRST SEARCH 
# ===============================
print("\n--- BEST-FIRST SEARCH ---")

solution_bfs = best_first_search(problem, f)

if solution_bfs:
    path_bfs = []
    current = solution_bfs
    while current:
        path_bfs.append(current.state)
        current = current.parent
    path_bfs.reverse()
    
    print("Solucion encontrada!")
    print(f"Camino: {' -> '.join(path_bfs)}")
    print(f"Costo total: {solution_bfs.path_cost}")
    print(f"Nodos explorados: {solution_bfs.nodes_explored}")
    
    print("\nOrden de exploracion:")
    for i, step in enumerate(solution_bfs.exploration_order[:8]):  # Solo primeros 8
        print(f"{i+1:2}. {step['state']} (costo: {step['path_cost']})")
        
else:
    print("No se encontro solucion")

# ===============================
# A* SEARCH 
# ===============================
print("\n--- A* SEARCH ---")

solution_astar = a_star_search(problem, h)

if solution_astar:
    path_astar = []
    current = solution_astar
    while current:
        path_astar.append(current.state)
        current = current.parent
    path_astar.reverse()
    
    print("Solucion encontrada!")
    print(f"Camino: {' -> '.join(path_astar)}")
    print(f"Costo total: {solution_astar.path_cost}")
    print(f"Nodos explorados: {solution_astar.nodes_explored}")
    
    print("\nOrden de exploracion:")
    for i, step in enumerate(solution_astar.exploration_order[:8]):  # Solo primeros 8
        print(f"{i+1:2}. {step['state']} (g:{step['path_cost']} + h:{step['heuristic']} = {step['f_value']})")
        
else:
    print("No se encontro solucion")

# ===============================
# COMPARACION
# ===============================
print("\n--- COMPARACION ---")

if solution_bfs and solution_astar:
    print(f"Best-First exploro: {solution_bfs.nodes_explored} nodos")
    print(f"A* exploro: {solution_astar.nodes_explored} nodos")
    
    if solution_astar.nodes_explored < solution_bfs.nodes_explored:
        diferencia = solution_bfs.nodes_explored - solution_astar.nodes_explored
        print(f"A* fue mas eficiente por {diferencia} nodos")
    elif solution_astar.nodes_explored > solution_bfs.nodes_explored:
        diferencia = solution_astar.nodes_explored - solution_bfs.nodes_explored
        print(f"Best-First fue mas eficiente por {diferencia} nodos")
    else:
        print("Ambos exploraron la misma cantidad de nodos")
    
    print(f"\nAmbos encontraron el mismo costo optimo: {solution_bfs.path_cost}")
    
    # Mostrar diferencia en estrategia
    print(f"\nPrimeras ciudades exploradas:")
    print(f"Best-First: {[step['state'] for step in solution_bfs.exploration_order[:5]]}")
    print(f"A*:         {[step['state'] for step in solution_astar.exploration_order[:5]]}")

# ===============================
# COMPARACIÓN DE RESULTADOS
# ===============================
print("\n" + "=" * 60)
print(" COMPARACIÓN DE RESULTADOS")
print("=" * 60)

if solution_bfs and solution_astar:
    print(f"Best-First Search - Costo: {solution_bfs.path_cost}")
    print(f"A* Search        - Costo: {solution_astar.path_cost}")
    
    if solution_astar.path_cost <= solution_bfs.path_cost:
        print("A* encontró una solución óptima (igual o mejor que Best-First)")
    else:
        print("Best-First encontró una mejor solución que A* (inusual)")
        
    print(f"\nLongitud del camino:")
    print(f"Best-First: {len(path_bfs)} ciudades")
    print(f"A*:         {len(path_astar)} ciudades")