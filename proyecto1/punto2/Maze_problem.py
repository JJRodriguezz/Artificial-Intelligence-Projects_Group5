import heapq # El módulo heapq implementa colas de prioridad (heaps)

class Node: # Nodo que representa cada estado en la búsqueda
    def __init__(self, position, parent=None, action=None, path_cost=0):
        self.position = position
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
    def __lt__(self, other):
        return self.path_cost < other.path_cost

class Problem: # Clase que modela el problema del laberinto
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.columns = len(maze[0]) if maze else 0
        
        self.actions = {"Up":(-1, 0), "Down":( 1, 0), "Left":( 0,-1), "Right":( 0, 1)} # Acciones posibles
        
        
        self.start = self.find_symbol("S") # Buscar inicio
        self.end  = self.find_symbol("E")  # Buscar fin

    def find_symbol(self, symbol): # Buscar símbolo en el laberinto
        for i in range(self.rows):
            for j in range(self.columns):
                if self.maze[i][j] == symbol:
                    return (i, j)
        return None
    
    
    def in_bounds(self, pos): # Verificar límites
        row, column = pos
        return 0 <= row < self.rows and 0 <= column < self.columns
    
    def passable(self, pos): # Verificar si se puede pasar por una casilla
        row, column = pos
        return self.maze[row][column] != '#'

# Algoritmo principal de búsqueda A* con heurística Manhattan
def find_exit(maze):

    problem = Problem(maze)  # Inicializa el problema
    start = problem.start    # Posición inicial
    end = problem.end        # Posición final

    def manhattan_distance(pos, end): # Heurística Manhattan
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    
    def get_neighbors(pos): # Obtener vecinos de cada posición
        neighbors = []
        for move in [x for x in problem.actions.keys()]:
            neighbor = (pos[0] + problem.actions[move][0], pos[1] + problem.actions[move][1])
            if problem.in_bounds(neighbor) and maze[neighbor[0]][neighbor[1]] != "#": # si el robot está en un borde y el vecino está es diferente a "#" pared agregarlo a la lista de vecinos
                neighbors.append((move, neighbor))
        return neighbors

    start_node = Node(start, path_cost=0)
    frontier = [(manhattan_distance(start, end), start_node)]
    heapq.heapify(frontier)
    reached = {start: start_node}

    while frontier: # Ciclo principal de búsqueda
        _, node = heapq.heappop(frontier)
        if node.position == end:
            return reconstruct_path(node)
        for action, neighbor_pos in get_neighbors(node.position):
            new_cost = node.path_cost + 1
            if neighbor_pos not in reached or new_cost < reached[neighbor_pos].path_cost:
                child = Node(neighbor_pos, parent=node, action=action, path_cost=new_cost)
                reached[neighbor_pos] = child
                priority = new_cost + manhattan_distance(neighbor_pos, end)
                heapq.heappush(frontier, (priority, child))
    return None

def reconstruct_path(node): # Reconstruye el camino y las acciones desde el nodo final
    
    path = []
    actions = []
    while node:
        path.append(node.position)
        actions.append(node.action)
        node = node.parent
    path.reverse()
    actions.reverse()
    if actions and actions[0] is None:
        actions = actions[1:]
    return path, actions

# Ejemplo de uso

maze = [["#", "S", "#", "#", "#", "#", "#", "#"],
        ["#", " ", "#", " ", "#", " ", " ", "E"],
        ["#", " ", " ", " ", "#", " ", " ", "#"],
        ["#", " ", "#", " ", " ", " ", "#", "#"],
        ["#", "#", "#", "#", "#", "#", "#", "#"],
        ["#", "#", "#", "#", "#", "#", "#", "#"]]

# Ejecutar búsqueda y mostrar los resultados obtenidos
path, actions = find_exit(maze)
print("Path to exit:", path)
print("Actions:", actions)

# Funciona 👍