import sys
from typing import List, Dict, Tuple

sys.setrecursionlimit(20000)

def solve_fallout_hash(grid: List[List[str]], device: str = 'cpu') -> int:
    if device == 'gpu':
        print("INFO: Implementación Hash no se beneficia de GPU. Ejecutando en CPU.")
        
    n = len(grid)
    max_steps = 2 * n - 2
    
    memo: Dict[Tuple[int, int, int], int] = {}
    
    result = _solve_recursive(0, 0, 0, n, max_steps, grid, memo)
    
    return result if result >= 0 else -1

def _solve_recursive(
    x: int, 
    y: int, 
    t: int, 
    n: int, 
    max_steps: int, 
    grid: List[List[str]], 
    memo: Dict[Tuple[int, int, int], int]
) -> int:
    state = (x, y, t)

    
    # 1. Fuera de los límites
    if not (0 <= x < n and 0 <= y < n):
        return -1 # Invalido

    # 2. Celda con bomba
    if grid[x][y] == 'B':
        return -1 # Invalido

    # 3. Límite de pasos excedido
    if t > max_steps:
        return -1 # Invalido
        
    # 4. Resultado ya calculado (Memoización)
    if state in memo:
        return memo[state]

    # 5. Se ha alcanzado la salida
    if x == n - 1 and y == n - 1:
        return 1 if grid[x][y] == 'R' else 0

    moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    
    max_from_neighbors = -1
    for next_x, next_y in moves:
        res = _solve_recursive(next_x, next_y, t + 1, n, max_steps, grid, memo)
        if res > max_from_neighbors:
            max_from_neighbors = res

    # Si no hay un camino válido desde los vecinos, esta ruta es inválida
    if max_from_neighbors == -1:
        memo[state] = -1
        return -1

    current_capsules = 1 if grid[x][y] == 'R' else 0
    total = current_capsules + max_from_neighbors
    
    memo[state] = total
    return total

if __name__ == '__main__':
    print("Ejecutando prueba unitaria para fallout_dp_hash.py...")
    
    test_grid = [
        ['S', 'R', '.'],
        ['.', 'B', 'R'],
        ['.', '.', 'E']
    ]
    
    test_grid_formatted = [
        ['.', 'R', '.'],
        ['.', 'B', 'R'],
        ['.', '.', '.']
    ]

    print("\n--- Prueba en CPU ---")
    expected = 2
    result = solve_fallout_hash(test_grid_formatted)
    print(f"Cuadrícula de prueba:\n{test_grid}")
    print(f"Resultado obtenido: {result}")
    print(f"Resultado esperado: {expected}")
    assert result == expected
    print("Prueba superada.")