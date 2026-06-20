# Parallel 2D Discrete Fourier Transform

> ID5130 — Parallel Scientific Computing · IIT Madras · Jan–May 2024 · Individual

Implemented the 2D Discrete Fourier Transform entirely from scratch in C/C++ — no FFTW, no NumPy — then parallelized it with OpenMP (shared memory) and MPI (distributed memory).

## What's Here

```
Parallel_DFT/
├── OMP.cpp                 ← OpenMP implementation
├── MPI.cpp                 ← MPI implementation
├── helper.ipynb            ← Analysis and plotting notebook
├── MPI_result.csv          ← MPI benchmark results
├── MPI_signal.csv          ← N×N random signal array consumed by MPI.cpp (N divisible by nprocs)
├── OMP_signal.csv          ← Grayscale image signal consumed by OMP.cpp
├── Result.png              ← Speedup and efficiency plots
├── sample_image.jpg        ← Test image for 2D DFT
└── ID5130_project.pdf      ← Full report
```

## Background

The Discrete Fourier Transform (DFT) converts a signal from the time/spatial domain to the frequency domain:

```
X[k] = Σ_{n=0}^{N-1} x[n] · e^{-2πi·kn/N}
```

For a 2D image of size M×N, the naive DFT is O(M²N²) — impractically slow for large images. This project implements the full 2D DFT naively (not FFT), then accelerates it through parallelism.

## Implementations

### OpenMP (Shared Memory)
- Parallelizes the outer loop over frequency components
- Uses `#pragma omp parallel for` with thread-local accumulators
- Thread count tuned to hardware core count
- Near-linear speedup up to ~8 threads

### MPI (Distributed Memory)
- Distributes rows of the 2D DFT across MPI processes
- Each process computes its assigned rows independently
- `MPI_Gather` collects results at root
- Works across multiple nodes (tested on IIT Madras cluster)

## Data Generation

`helper.ipynb` prepares the inputs for both implementations and visualizes results:
- For OpenMP: converts `sample_image.jpg` to grayscale and exports it as `OMP_signal.csv`
- For MPI: generates an N×N array of random values (0–255), with N divisible by the number of processes, exported as `MPI_signal.csv`
- Also includes cells for verifying the conversion and visualizing signals/results

Each benchmark configuration (`OMP.cpp` and `MPI.cpp`) was run 10 times; results were averaged and the values recorded in `helper.ipynb` for the plots below.

## Results

| Threads / Processes | Speedup | Efficiency |
|---|---|---|
| 1 (baseline) | 1.0× | 100% |
| 2 | 1.97× | 98.5% |
| 4 | 3.89× | 97.2% |
| 8 | 7.71× | 96.4% |
| 16 | 14.8× | 92.5% |

**>80% efficiency** maintained across all tested thread counts.  
**Near-linear speedup** with 1:1 thread-to-speedup correspondence up to 8 threads.

## How to Run

**OpenMP:**
```bash
g++ -O2 -fopenmp -o dft_omp OMP.cpp
OMP_NUM_THREADS=8 ./dft_omp sample_image.jpg
```

**MPI:**
```bash
mpicxx -O2 -o dft_mpi MPI.cpp
mpirun -np 8 ./dft_mpi sample_image.jpg
```

## Technologies

`C++17` · `OpenMP` · `MPI (OpenMPI)` · `Python + Matplotlib` (analysis)
