import argparse
import time
import random
import os
import sys
from collections import defaultdict

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import psutil
except ImportError as e:
    print(f"Error: Falta una dependencia. Por favor, instala '{e.name}' usando pip.")
    sys.exit(1)

from fallout_dp_array import solve_fallout_array
from fallout_dp_hash import solve_fallout_hash

def generate_grid(n: int, prob_bomb: float, prob_capsule: float, seed: int) -> list[list[str]]:
    random.seed(seed)
    grid = [['.' for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            # Las celdas de inicio y fin no pueden ser bombas
            if (r == 0 and c == 0) or (r == n - 1 and c == n - 1):
                continue
            
            rand_val = random.random()
            if rand_val < prob_bomb:
                grid[r][c] = 'B'
            elif rand_val < prob_bomb + prob_capsule:
                grid[r][c] = 'R'
    return grid

def run_benchmark(sizes: list[int], trials: int, device: str, seed: int):
    print(f"--- Iniciando Benchmark ---")
    print(f"Dispositivo: {device.upper()} | Tamaños: {sizes} | Pruebas por tamaño: {trials}\n")

    results = defaultdict(lambda: defaultdict(list))
    process = psutil.Process(os.getpid())

    for n in sizes:
        print(f"--- Procesando tamaño n={n} ---")
        for i in range(trials):
            current_seed = seed + n * trials + i
            grid = generate_grid(n, prob_bomb=0.15, prob_capsule=0.3, seed=current_seed)
            
            mem_before = process.memory_info().rss
            start_time = time.perf_counter()
            solve_fallout_array(grid, device)
            end_time = time.perf_counter()
            mem_after = process.memory_info().rss
            
            results[n]['array_time'].append(end_time - start_time)
            results[n]['array_mem'].append(mem_after - mem_before)
            
            mem_before = process.memory_info().rss
            start_time = time.perf_counter()
            solve_fallout_hash(grid, device)
            end_time = time.perf_counter()
            mem_after = process.memory_info().rss
            
            results[n]['hash_time'].append(end_time - start_time)
            results[n]['hash_mem'].append(mem_after - mem_before)

            print(f"  Trial {i+1}/{trials} | "
                  f"Array: {results[n]['array_time'][-1]:.4f}s, "
                  f"Hash: {results[n]['hash_time'][-1]:.4f}s")
    
    avg_results = defaultdict(dict)
    for n in sizes:
        for metric in ['array_time', 'array_mem', 'hash_time', 'hash_mem']:
            avg_results[n][metric] = np.mean(results[n][metric])

    print("\n--- Resultados Promedio ---")
    print("Tamaño | T. Array (s) | Mem. Array (MB) | T. Hash (s) | Mem. Hash (MB)")
    print("-" * 70)
    for n in sorted(avg_results.keys()):
        ar = avg_results[n]
        print(f"{n:<6} | "
              f"{ar['array_time']:<12.4f} | "
              f"{ar['array_mem']/1e6:<15.4f} | "
              f"{ar['hash_time']:<11.4f} | "
              f"{ar['hash_mem']/1e6:<14.4f}")

    plot_results(avg_results, device)

def plot_results(avg_results: dict, device: str):
    sizes = sorted(avg_results.keys())
    array_times = [avg_results[n]['array_time'] for n in sizes]
    hash_times = [avg_results[n]['hash_time'] for n in sizes]
    array_mems = [avg_results[n]['array_mem'] / 1e6 for n in sizes] # en MB
    hash_mems = [avg_results[n]['hash_mem'] / 1e6 for n in sizes] # en MB

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, array_times, 'o-', label=f'Array 3D ({device.upper()})', color='crimson')
    ax.plot(sizes, hash_times, 's-', label=f'Hash Dict (CPU)', color='royalblue')
    ax.set_xlabel("Tamaño de la Cuadrícula (n)")
    ax.set_ylabel("Tiempo de Ejecución Promedio (s)")
    ax.set_title(f"Tiempo de Ejecución vs. Tamaño de Cuadrícula ({device.upper()})")
    ax.set_xticks(sizes)
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig("benchmark_time.png")
    print("\nGráfica de tiempo guardada en 'benchmark_time.png'")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, array_mems, 'o-', label=f'Array 3D ({device.upper()})', color='crimson')
    ax.plot(sizes, hash_mems, 's-', label=f'Hash Dict (CPU)', color='royalblue')
    ax.set_xlabel("Tamaño de la Cuadrícula (n)")
    ax.set_ylabel("Uso de Memoria Adicional Promedio (MB)")
    ax.set_title(f"Uso de Memoria vs. Tamaño de Cuadrícula ({device.upper()})")
    ax.set_xticks(sizes)
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig("benchmark_memory.png")
    print("Gráfica de memoria guardada en 'benchmark_memory.png'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark para solucionadores de Fallout DP.")
    parser.add_argument('--sizes', nargs='+', type=int, default=[10, 15, 20],
                        help='Lista de tamaños de cuadrícula (n) a probar.')
    parser.add_argument('--trials', type=int, default=3,
                        help='Número de pruebas a ejecutar por cada tamaño.')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help="Dispositivo a usar para la implementación con array ('cpu' o 'gpu').")
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria base para la generación de cuadrículas.')

    args = parser.parse_args()
    
    if args.device == 'gpu':
        try:
            import cupy
        except ImportError:
            print("ERROR: CuPy no está instalado. No se puede ejecutar en modo GPU.")
            print("Por favor, instale CuPy (ej: pip install cupy-cuda11x) o use --device cpu.")
            sys.exit(1)

    run_benchmark(args.sizes, args.trials, args.device, args.seed)