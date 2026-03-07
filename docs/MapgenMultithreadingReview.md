# Map Generation Multithreading Review

## Current Pipeline Hotspots
- Biome sampling over full map (`generate_biome_terrain` base tile fill loop).
- Forest density generation and Gaussian blur passes.
- Dirt/stone/shore per-cell neighborhood scans.

## Multithreading Feasibility (Current Pure Python)
- CPU-heavy loops are mostly Python-level numeric loops.
- Python threads will be constrained by the GIL for these loops.
- Expected gain from `ThreadPoolExecutor` on current code: low or negative (overhead + contention).

## Where Parallelism Helps
- Parallelizing *multiple whole-map generations* (e.g. dataset creation) across processes.
- Offloading per-cell math to vectorized native kernels (NumPy/SciPy), which release GIL and can use multithreaded BLAS/SIMD.
- Compiling hotspot loops (Numba/Cython/Rust extension) before applying threads.

## Recommended Path
1. Keep current single-map generation single-threaded in pure Python.
2. Add optional vectorized mask/blur implementation using NumPy for:
   - forest mask accumulation
   - Gaussian blur
   - shoreline neighbor counts
3. For bulk generation, use process-level parallelism (`multiprocessing` / worker pool), one map per process.
4. If runtime still high, move hotspot kernels to Numba/Cython and then benchmark threaded chunking.

## Safe Immediate Optimizations (No Threading)
- Reuse scratch arrays for masks/blur buffers between passes.
- Reduce repeated dict lookups inside inner loops by hoisting local references.
- Use integer tile classes for temporary computation, convert to tile IDs at end.

## Benchmark Guidance
- Benchmark on fixed seed and map size buckets (64, 96, 128).
- Report p50/p95 generation time and memory.
- Compare:
  - baseline pure Python
  - vectorized (NumPy) single-thread
  - process-parallel batch generation
