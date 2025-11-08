# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ optimization project for AGV (Automated Guided Vehicle) charging scheduling and task allocation. The project implements multiple solution approaches including exact optimization with Gurobi, heuristic algorithms (ALNS), and Logic-based Benders Decomposition (LBBD).

## Build System

### Dependencies
- **Gurobi Optimizer**: Required for mathematical optimization models
  - Environment variable `GUROBI_HOME` must be set
  - FindGUROBI.cmake handles library detection
- **OpenMP**: Required for parallel processing
- **CMake**: Minimum version 3.22
- **C++17**: Required standard

### Build Commands

**Debug Build:**
```bash
mkdir cmake-build-debug
cd cmake-build-debug
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Debug
```

**Release Build:**
```bash
mkdir cmake-build-release
cd cmake-build-release
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
```

**Run Executable:**
```bash
# Debug version
./cmake-build-debug/Debug/predict_agv_charging.exe

# Release version
./cmake-build-release/Release/predict_agv_charging.exe
```

## Code Architecture

### Core Components

1. **Model**: Main optimization model class
   - Supports multiple solving methods: Gurobi exact, ALNS heuristic, LBBD
   - Manages master and subproblem models for decomposition approaches

2. **Parameter**: Configuration and data management
   - Handles test cases, distance matrices, AGV/task parameters
   - Manages uncertainty scenarios for robust optimization
   - Supports multiple model types: DETERMINISTIC, SAA, DRO

3. **AGV**: Represents individual automated guided vehicles
   - Contains task sequences, charging schedules, SOC (State of Charge) tracking
   - Implements schedule validation and time calculations

4. **Task**: Individual container handling operations
   - Defined in Task.h/cpp with location and timing information

5. **TestUnit**: Test framework for running experiments
   - Configured with solver method, model type, and test parameters
   - Handles output redirection and result collection

### Heuristic Framework (TruckHeuristic/)

ALNS (Adaptive Large Neighborhood Search) implementation:
- **Destroy Operators**: Remove tasks/charging decisions from solution
  - Charging-based: Critical/Random/Worst removal
  - Station-based: Critical/Random/Worst removal
- **Repair Operators**: Reconstruct feasible solutions
  - Greedy/Adaptive repair with Random/Time/Worst strategies
- **Local Search**: Neighborhood-based improvements
- **Simulated Annealing**: Acceptance criterion with cooling schedules

### Solution Methods

1. **GUROBI**: Exact mathematical programming
2. **HEURISTICS**: ALNS metaheuristic
3. **LBBD_GUROBI**: Logic-based Benders with Gurobi subproblems
4. **LBBD_RULE**: Logic-based Benders with rule-based subproblem solving

## Key Files

- `main.cpp`: Entry point - creates and runs TestUnit
- `Model.h/cpp`: Core optimization model
- `Parameter.h/cpp`: Problem data and configuration
- `AGV.h/cpp`: Vehicle representation and scheduling
- `GurobiModel.h/cpp`: Gurobi-specific model implementation
- `BendersCut.h/cpp`: Benders decomposition cuts
- `TestUnit.h/cpp`: Experimental framework
- `distance_matrix.csv`: Precomputed travel distances
- `CMakeLists.txt`: Build configuration
- `FindGUROBI.cmake`: Gurobi library detection

## Testing

The project uses a custom TestUnit framework. Modify `main.cpp` to configure test scenarios:

```cpp
TestUnit test_exact_DRO2("small", SolveMethod::HEURISTICS, true, ModelType::DETERMINISTIC, true);
```

Parameters: test_type, solve_method, warm_start, model_type, output_to_file

### Data Files

- `distance_matrix.csv`: Precomputed travel distances between locations
- `AGV充电时间预测_所有区间_随机森林.csv/xlsx`: AGV charging time prediction data
- `random_seed.h`: Random number generation utilities

## Algorithm Documentation

The README.md contains detailed algorithm documentation including:
- Dynamic LCFS (Last Critical-path First Serve) algorithm implementation
- Subproblem solving with conflict resolution strategies
- Step-by-step algorithm flow and examples
- Theoretical foundations and optimality guarantees