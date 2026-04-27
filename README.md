# Terrain Generation Benchmark

Performance comparison of Perlin noise heightmap generation across different execution models: sequential CPU, parallel CPU (Parlay), CUDA GPU, and a hybrid approach that splits work between both.

## What this is

A benchmarking tool that generates terrain heightmaps using Perlin noise and measures how fast each execution model runs. The goal is to see where parallelization and GPU acceleration help (and where they don't).

Four modes are available:

| Flag | Mode |
|------|------|
| `-s` | Sequential CPU |
| `-p` | Parallel CPU (Parlay) |
| `-c` | CUDA GPU |
| `-h` | Hybrid (split work between CPU + GPU) |

## Building

Requires CMake 3.15+, a C++17 compiler, CUDA toolkit, and [glm](https://github.com/g-truc/glm). Parlay and modernGPU are vendored under `include/`.

```bash
cmake --preset default
cmake --build --preset release
```

The binary lands at `build/benchmark_world_gen`.

## Running

Basic usage:

```bash
# Parallel CPU, 2048x2048 grid (default mode)
./build/benchmark_world_gen -dim 2048

# CUDA mode with a specific seed
./build/benchmark_world_gen -c -dim 1024 -seed 1337
```

Additional flags:

| Flag | Description |
|------|-------------|
| `-dim N` | Grid size (default 256) |
| `-seed N` | Random seed (default 42) |
| `-hybrid_gen_split F` | Fraction of generation on GPU for hybrid mode (default 0.95) |
| `-hybrid_norm_split F` | Fraction of normalization on GPU for hybrid mode (default 0.95) |

Each mode runs 10 iterations and reports timing. A 3-second warmup runs first to let the CPU/GPU reach steady state.

## Scripts

Three helper scripts live in the repo root:

- **`run_benchmarks.sh`** -- Runs sequential vs parallel CPU across multiple dimensions (1024–8192) and seeds. Outputs `benchmark_results.csv`.
- **`run_cuda_bench.sh`** -- Compares parallel CPU vs CUDA at increasing grid sizes up to 16384x16384. Outputs `benchmark_cpu_vs_cuda.csv`.
- **`plot_graph.py`** -- Reads the CSV from `run_cuda_bench.sh` and plots average speedup (CPU / CUDA) as a function of grid dimension. Saves to `speedup.png`.

## Structure

```
src/
  perlin_noise_cpu.hpp    -- Sequential + parallel CPU Perlin noise (Parlay)
  perlin_noise_cuda.cu    -- Pure CUDA implementation
  perlin_noise_hybrid.cu  -- Hybrid CPU/GPU split
  heightmap_gen.hpp       -- MapGenerator class with all four generation paths
  benchmark.hpp           -- Benchmark harness and timer logic
  cmd_parser.hpp          -- Command-line argument parsing
include/                  -- Vendored: glm, parlay, moderngpu
```
