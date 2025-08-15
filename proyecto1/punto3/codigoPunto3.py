import time
import tracemalloc
from collections import deque
from typing import List, Dict, Optional, Tuple

class Node: #clase que representa el nodo en el grafo

    def __init__(self, state: str, path: List[str] = None, depth: int = 0, parent=None):
        self.state = state  # Estación actual
        self.path = path if path is not None else [state]  # Camino recorrido
        self.depth = depth  # Profundidad del nodo
        self.parent = parent  # Nodo padre
    
    def __str__(self):
        return f"Node(state={self.state}, depth={self.depth}, path={self.path})"
    
    def __repr__(self):
        return self.__str__()

class MetroProblem: #Grafo que contiene las estaciones y conexiones del metro

    def __init__(self):
        # Definición del grafo de la red de metro
        self.graph = {
            'A': ['B', 'C'],
            'B': ['A', 'D', 'E'],
            'C': ['A', 'F'],
            'D': ['B', 'G'],
            'E': ['B', 'H', 'I'],
            'F': ['C', 'J'],
            'G': ['D'],
            'H': ['E'],
            'I': ['E', 'J'],
            'J': ['F', 'I']
        }

    def actions(self, state: str) -> List[str]: #devuelve las conexiones o acciones posibles desde una estación dada

        return self.graph.get(state, [])
    
    def result(self, state: str, action: str) -> str: #devuelve el estado despues de hacer una acción

        if action in self.actions(state):
            return action
        return None
    
    def goal_test(self, state: str, goal: str) -> bool: #revisa si el estado en el que esta es el objetivo (si la estacion en la que se encuentra es la final)

        return state == goal
    
    def display_graph(self): #muestra la representacion del grafo
        
        print("Red de Metro:")
        for station, connections in self.graph.items():
            print(f"Estación {station} conectada a: {', '.join(connections)}")

class SearchAlgorithms: #aca estan los dos algoritmos de busqueda (BFS e IDS)

    def __init__(self, problem: MetroProblem):
        self.problem = problem
        self.nodes_explored = 0
        self.max_frontier_size = 0
    
    def breadth_first_search(self, initial: str, goal: str) -> Tuple[Optional[List[str]], Dict]: #implementacin de BFS
        #aca explora por niveles hasta encontrar la solucion

        print(f"\n=== BREADTH-FIRST SEARCH ===")
        print(f"Buscando ruta de {initial} a {goal}")
        
        # Reiniciar los contadores
        self.nodes_explored = 0
        self.max_frontier_size = 0
        
        # Aca tambien se revisa si el estado en el que esta es el final u objetivo
        if self.problem.goal_test(initial, goal):
            return [initial], {"nodes_explored": 0, "max_frontier_size": 0}
        
        # Inicializa la frontera con el nodo inicial
        frontier = deque([Node(initial)])
        explored = set()
        
        while frontier:
            # Actualiza tamaño máximo de la frontera
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            
            # Extrae nodo de la frontera
            node = frontier.popleft()
            
            # Marca como explorado
            explored.add(node.state)
            self.nodes_explored += 1
            
            print(f"Explorando: {node.state} (profundidad {node.depth})")
            
            # Explorar acciones posibles
            for action in self.problem.actions(node.state):
                child_state = self.problem.result(node.state, action)
                
                # Verifica si el hijo no ha sido explorado ni está en la frontera
                if child_state not in explored and not any(n.state == child_state for n in frontier):
                    child_path = node.path + [child_state]
                    child = Node(child_state, child_path, node.depth + 1, node)
                    
                    # Verifica si es el objetivo
                    if self.problem.goal_test(child_state, goal):
                        print(f"¡Meta encontrada! Ruta: {' -> '.join(child.path)}")
                        stats = {
                            "nodes_explored": self.nodes_explored + 1,
                            "max_frontier_size": self.max_frontier_size,
                            "solution_depth": child.depth
                        }
                        return child.path, stats
                    
                    frontier.append(child)
        
        return None, {"nodes_explored": self.nodes_explored, "max_frontier_size": self.max_frontier_size}
    
    def iterative_deepening_search(self, initial: str, goal: str, max_depth: int = 10) -> Tuple[Optional[List[str]], Dict]:
        #implementacion del IDS donde se combina DFS con busqueda en anchura aumentando la profundidad

        print(f"\n=== ITERATIVE DEEPENING SEARCH ===")
        print(f"Buscando ruta de {initial} a {goal}")
        
        total_nodes_explored = 0
        max_frontier_size = 0
        
        for depth_limit in range(max_depth + 1):
            print(f"\nProfundidad límite: {depth_limit}")
            
            # Reinicia contadores para esta iteración
            self.nodes_explored = 0
            self.max_frontier_size = 0
            
            result, stats = self._depth_limited_search(initial, goal, depth_limit)
            
            total_nodes_explored += self.nodes_explored
            max_frontier_size = max(max_frontier_size, self.max_frontier_size)
            
            if result is not None:
                print(f"¡Meta encontrada en profundidad {depth_limit}! Ruta: {' -> '.join(result)}")
                final_stats = {
                    "nodes_explored": total_nodes_explored,
                    "max_frontier_size": max_frontier_size,
                    "solution_depth": len(result) - 1,
                    "depth_iterations": depth_limit + 1
                }
                return result, final_stats
            
            print(f"No encontrado en profundidad {depth_limit}. Nodos explorados: {self.nodes_explored}")
        
        return None, {
            "nodes_explored": total_nodes_explored, 
            "max_frontier_size": max_frontier_size,
            "depth_iterations": max_depth + 1
        }
    
    def _depth_limited_search(self, initial: str, goal: str, depth_limit: int) -> Tuple[Optional[List[str]], Dict]:
        # Limitar la profundidad para la búsqueda
        return self._recursive_dls(Node(initial), goal, depth_limit)
    
    def _recursive_dls(self, node: Node, goal: str, depth_limit: int) -> Tuple[Optional[List[str]], Dict]:
        #implementacion de la busqueda en profundidad limitada

        self.nodes_explored += 1
        
        # Verificar si es el objetivo
        if self.problem.goal_test(node.state, goal):
            return node.path, {}
        
        # Si alcanzamos el límite de profundidad, retornar None
        if node.depth >= depth_limit:
            return None, {}
        
        # Explorar hijos
        for action in self.problem.actions(node.state):
            child_state = self.problem.result(node.state, action)
            
            # Evitar ciclos verificando si el estado no está en el camino actual
            if child_state not in node.path:
                child_path = node.path + [child_state]
                child = Node(child_state, child_path, node.depth + 1, node)
                
                result, _ = self._recursive_dls(child, goal, depth_limit)
                if result is not None:
                    return result, {}
        
        return None, {}

def performance_comparison(): #aca se ejecutan los dos algoritmos y se compara su rendimiento
    # Crear problema e instancia de algoritmos
    problem = MetroProblem()
    search = SearchAlgorithms(problem)
    
    # Mostrar el grafo
    problem.display_graph()
    
    print("\n" + "="*60)
    print("COMPARACIÓN DE ALGORITMOS: BFS vs IDS")
    print("Ruta: Estación A → Estación J")
    print("="*60)
    
    # Ejecutar BFS
    print("\n EJECUTANDO BFS...")
    tracemalloc.start()
    start_time = time.time()
    
    bfs_result, bfs_stats = search.breadth_first_search('A', 'J')
    
    bfs_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    bfs_memory = peak
    tracemalloc.stop()
    
    # Ejecutar IDS
    print("\n EJECUTANDO IDS...")
    tracemalloc.start()
    start_time = time.time()
    
    ids_result, ids_stats = search.iterative_deepening_search('A', 'J')
    
    ids_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    ids_memory = peak
    tracemalloc.stop()
    
    # Mostrar resultados
    print("\n" + "="*60)
    print(" RESULTADOS DE LA COMPARACIÓN")
    print("="*60)
    

    
    print(f"\n BFS (Breadth-First Search):")
    print(f"   Ruta encontrada: {' → '.join(bfs_result) if bfs_result else 'No encontrada'}")
    print(f"   Longitud de la ruta: {len(bfs_result) - 1 if bfs_result else 'N/A'} paradas")
    print(f"   Nodos explorados: {bfs_stats['nodes_explored']}")
    print(f"   Tamaño máximo de frontera: {bfs_stats['max_frontier_size']}")
    print(f"   Tiempo de ejecución: {bfs_time:.6f} segundos")
    print(f"   Memoria utilizada: {bfs_memory / 1024:.2f} KB")
    
    print("="*60)
    
    print(f"\n IDS (Iterative Deepening Search):")
    print(f"   Ruta encontrada: {' → '.join(ids_result) if ids_result else 'No encontrada'}")
    print(f"   Longitud de la ruta: {len(ids_result) - 1 if ids_result else 'N/A'} paradas")
    print(f"   Nodos explorados: {ids_stats['nodes_explored']}")
    print(f"   Tamaño máximo de frontera: {ids_stats['max_frontier_size']}")
    print(f"   Iteraciones de profundidad: {ids_stats.get('depth_iterations', 'N/A')}")
    print(f"   Tiempo de ejecución: {ids_time:.6f} segundos")
    print(f"   Memoria utilizada: {ids_memory / 1024:.2f} KB")

    print("="*60)

    # Análisis comparativo
    print(f"\n ANÁLISIS COMPARATIVO:")
    print(f"   Eficiencia temporal: {'BFS' if bfs_time < ids_time else 'IDS'} fue más rápido")
    print(f"   Diferencia de tiempo: {abs(bfs_time - ids_time):.6f} segundos")
    print(f"   Eficiencia espacial: {'BFS' if bfs_memory < ids_memory else 'IDS'} utilizó menos memoria")
    print(f"   Diferencia de memoria: {abs(bfs_memory - ids_memory) / 1024:.2f} KB")
    print(f"   Exploración de nodos: {'BFS' if bfs_stats['nodes_explored'] < ids_stats['nodes_explored'] else 'IDS'} exploró menos nodos")
    
    return {
        'bfs': {'result': bfs_result, 'stats': bfs_stats, 'time': bfs_time, 'memory': bfs_memory},
        'ids': {'result': ids_result, 'stats': ids_stats, 'time': ids_time, 'memory': ids_memory}
    }

    
    
def analyze_different_routes(): #se hacen ejemplos analizando otras rutas distintas a A -> J

    problem = MetroProblem()
    search = SearchAlgorithms(problem)
    
    test_routes = [
        ('A', 'J'),  # Ruta larga
        ('A', 'B'),  # Ruta corta
        ('G', 'H'),  # Ruta media
        ('C', 'I')   # Ruta alternativa
    ]
    
    print("\n" + "="*60)
    print(" ANÁLISIS DE DIFERENTES RUTAS")

    
    for start, goal in test_routes:
        print("="*60)
        print(f"\n  Analizando ruta: {start} → {goal}")
        
        # BFS
        bfs_result, bfs_stats = search.breadth_first_search(start, goal)
        
        # IDS
        ids_result, ids_stats = search.iterative_deepening_search(start, goal)
        
        print(f"   BFS: {' → '.join(bfs_result)} ({bfs_stats['nodes_explored']} nodos)")
        print(f"   IDS: {' → '.join(ids_result)} ({ids_stats['nodes_explored']} nodos)")

if __name__ == "__main__":
    # Ejecutar comparación principal
    results = performance_comparison()
    
    # Ejecutar análisis adicional
    analyze_different_routes()
    
    print("="*60)
    
    print(f"\n Análisis completado. Ambos algoritmos encontraron la ruta óptima de A a J.")
    print(f"   La ruta más corta requiere {len(results['bfs']['result']) - 1} paradas.")