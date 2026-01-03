# Iterative methods (Gauss-Seidel)

- **Student:** Zaharov Gleb Mihajlovič, group 3823Б1ПР4
- **Technology:** SEQ | MPI
- **Variant**: 19


## 1. Introduction
This project implements the Seidel method (Gauss-Seidel iterative method) for
solving systems of linear equations. The method is particularly suitable for
large sparse systems and can benefit from parallelization due to its iterative
nature. The goal is to compare sequential and parallel MPI implementations,
demonstrating the performance advantages of distributed computing for
large-scale numerical problems.


## 2. Problem Statement
Solve a system of linear equations **Ax = b** where:
- **A** is an n×n matrix with diagonal dominance: `A[i][i] = n + 1`, `A[i][j] = 1/(|i-j|+1)` for i ≠ j
- **b** is a vector: `b[i] = i + 1`
- **x** is the unknown solution vector

**Input**: Three values: system size (n > 0), precision (ε > 0), maximum iterations (max_iter > 0)
**Output**: Solution vector x of length n
**Constraints**: The matrix is guaranteed to be diagonally dominant, ensuring convergence.


## 3. Baseline Algorithm (Sequential)
The sequential Gauss-Seidel algorithm:

```py
Initialize x = [0, 0, ..., 0]
for iteration in 1..max_iterations:
  max_diff = 0

  for i in 0..n-1:
    sum = b[i]
    for j in 0..i-1:
      sum -= A[i][j] * x[j]  # Use updated values
    for j in i+1..n-1:
      sum -= A[i][j] * x[j]  # Use old values
    new_x = sum / A[i][i]
    max_diff = max(max_diff, |new_x - x[i]|)
    x[i] = new_x

  if max_diff < epsilon:
    break
```

Time complexity: O(max_iterations × n²) per iteration.


## 4. Parallelization Scheme

### MPI Implementation:
- **Data Distribution**: Matrix rows are distributed cyclically across processes
- **Communication Pattern**: All-to-all communication using `MPI_Allgatherv` after each iteration
- **Rank Roles**: All processes compute their assigned rows, then exchange results

**Process distribution**:
```
Process 0: rows 0, P, 2P, ...
Process 1: rows 1, P+1, 2P+1, ...
...
Process P-1: rows P-1, 2P-1, 3P-1, ...
```

**Pseudocode**:
```python
# Each process:
local_rows = assign_rows(rank, total_processes)
local_A = A[local_rows, :]  # Each process stores only its rows
local_b = b[local_rows]

while not converged and iterations < max_iter:
  compute_local_x(local_A, local_b, global_x)
  MPI_Allgatherv(local_x, global_x)  # Exchange results
  check_convergence(global_x, prev_x)
```


## 5. Implementation Details

### Code Structure
```
tasks/zaharov_g_seidel_int_met/
├── mpi/
│   ├── include/ops_mpi.hpp      # MPI interface
│   └── src/ops_mpi.cpp          # MPI implementation
├── seq/
│   ├── include/ops_seq.hpp      # Sequential interface
│   └── src/ops_seq.cpp          # Sequential implementation
└── tests/                       # Test suites
```

**Memory Usage**:
- Sequential: Stores full n×n matrix (O(n²) memory)
- MPI: Each process stores only its assigned rows (O(n²/P) memory)

**Assumptions**:
- Matrix is diagonally dominant (guarantees convergence)
- System size divisible by number of processes (with remainder handling)
- All processes participate in computation


## 6. Experimental Setup

### Hardware/Software Environment
- **CPU:** Intel Core i5-10600KF (6 cores, 12 threads, 4.1GHz base)
- **RAM:** 32GB
- **OS:** NixOS (Linux-based distribution)
- **Compiler:** GCC 14.3
- **Build Type:** Release

### Test Configuration
- **Matrix size:** 2000×2000
- **Number of Processes:** 6

## 7. Results and Discussion

### 7.1 Correctness
Correctness verified through:
- Residual norm calculation: `||Ax - b|| / n < ε × 1000`
- Comparison with sequential reference implementation
- Unit tests covering various system sizes (10, 50, 100)
- Boundary cases and error conditions

All functional tests pass for both SEQ and MPI implementations with identical results within tolerance.

### 7.2 Performance

#### Execution Times
Performance results for system size 2000, ε=1e-6, max_iterations=1500:

| Procs | Mode     | Implementation | Time, (s) | Speedup |
|-------|----------|----------------|-----------|---------|
| 1     | Task Run | SEQ            | 0.0063    | 1.00    |
| 1     | Task Run | MPI            | 0.0066    | 0.96    |
| 1     | Pipeline | SEQ            | 0.0235    | 1.00    |
| 1     | Pipeline | MPI            | 0.0506    | 0.46    |
| 2     | Task Run | SEQ            | 0.0065    | 1.00    |
| 2     | Task Run | MPI            | 0.0033    | 1.97    |
| 2     | Pipeline | SEQ            | 0.0262    | 1.00    |
| 2     | Pipeline | MPI            | 0.0339    | 0.77    |
| 4     | Task Run | SEQ            | 0.0073    | 1.00    |
| 4     | Task Run | MPI            | 0.0018    | 4.06    |
| 4     | Pipeline | SEQ            | 0.0308    | 1.00    |
| 4     | Pipeline | MPI            | 0.0307    | 1.00    |
| 6     | Task Run | SEQ            | 0.0118    | 1.00    |
| 6     | Task Run | MPI            | 0.0020    | 5.90    |
| 6     | Pipeline | SEQ            | 0.0446    | 1.00    |
| 6     | Pipeline | MPI            | 0.0366    | 1.22    |

#### Key Observations:
- **task_run** measures pure algorithm time: MPI shows 5.9× speedup with 6 processes
- **pipeline** includes initialization/communication overhead: 1.51× speedup

**Scalability Limitations**:
- Communication overhead grows with process count
- Load imbalance for non-divisible system sizes
- Memory access patterns affect cache efficiency
