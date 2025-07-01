import sys
from typing import List, Tuple

sys.setrecursionlimit(20000)

def solve_fallout_array(grid: List[List[str]], device: str = 'cpu') -> int:
    if device == 'gpu':
        try:
            import cupy as np
            print("INFO: Usando CuPy para cálculos en GPU.")
        except ImportError:
            print("ADVERTENCIA: CuPy no está instalado. Usando NumPy en CPU.")
            import numpy as np
    else:
        import numpy as np
        print("INFO: Usando NumPy para cálculos en CPU.")

    n = len(grid)
    max_steps = 2 * n - 2  

    grid_np = np.array([[1 if cell == 'R' else 0 for cell in row] for row in grid])
    bombs = np.array([[1 if cell == 'B' else 0 for cell in row] for row in grid])


    memo = np.full((n, n, max_steps + 1), -2, dtype=np.int32)

    result = _solve_recursive(0, 0, 0, n, max_steps, grid_np, bombs, memo, np)

    return int(result) if result >= 0 else -1

def _solve_recursive(
    x: int, 
    y: int, 
    t: int, 
    n: int, 
    max_steps: int, 
    grid_np, 
    bombs, 
    memo, 
    np
) -> int:
   
    if not (0 <= x < n and 0 <= y < n):
        return -1  # Invalido

    # 2. Celda con bomba
    if bombs[x, y] == 1:
        return -1  # Invalido

    # 3. Límite de pasos excedido
    if t > max_steps:
        return -1  # Invalido

    # 4. Resultado ya calculado (Memoización)
    if memo[x, y, t] != -2:
        return memo[x, y, t]

    # 5. Se ha alcanzado la salida
    if x == n - 1 and y == n - 1:
        # Devuelve el valor de la celda de salida
        return grid_np[x, y]

    
    moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    
    max_from_neighbors = -1
    for next_x, next_y in moves:
        res = _solve_recursive(next_x, next_y, t + 1, n, max_steps, grid_np, bombs, memo, np)
        if res > max_from_neighbors:
            max_from_neighbors = res

    # Si no hay un camino válido desde los vecinos, esta ruta es inválida
    if max_from_neighbors == -1:
        memo[x, y, t] = -1
        return -1

    # El total es lo de la celda actual más el máximo de los siguientes pasos
    current_capsules = grid_np[x, y]
    total = current_capsules + max_from_neighbors
    
    memo[x, y, t] = total
    return total

if __name__ == '__main__':
    print("Ejecutando prueba unitaria para fallout_dp_array.py...")
    
    
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
    expected_cpu = 2 
    result_cpu = solve_fallout_array(test_grid_formatted, device='cpu')
    print(f"Cuadrícula de prueba:\n{test_grid}")
    print(f"Resultado obtenido (CPU): {result_cpu}")
    print(f"Resultado esperado: {expected_cpu}")
    assert result_cpu == expected_cpu
    print("Prueba en CPU superada.")

    # Prueba con GPU
    try:
        import cupy
        print("\n--- Prueba en GPU ---")
        expected_gpu = 2
        result_gpu = solve_fallout_array(test_grid_formatted, device='gpu')
        print(f"Resultado obtenido (GPU): {result_gpu}")
        print(f"Resultado esperado: {expected_gpu}")
        assert result_gpu == expected_gpu
        print("Prueba en GPU superada.")
    except ImportError:
        print("\nADVERTENCIA: CuPy no encontrado, omitiendo prueba de GPU.")