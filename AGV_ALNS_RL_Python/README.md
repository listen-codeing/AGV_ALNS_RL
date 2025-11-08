# AGV Charging Scheduling with RL-enhanced ALNS

Python implementation of Adaptive Large Neighborhood Search (ALNS) with Reinforcement Learning for solving the AGV (Automated Guided Vehicle) charging scheduling problem.

## 🎯 Overview

This project converts the original C++ AGV charging problem solver to Python and enhances it with:

1. **ALNS Framework** (adapted from alns-framework-for-evsp)
   - 6 destroy operators (charging-based and station-based removal)
   - 9 repair operators (3×3 strategy matrix: Greedy/Random/Adaptive × Best/Random/Time)
   - Adaptive weight management with roulette wheel selection
   - Multiple acceptance criteria (Simulated Annealing, Great Deluge, Hill Climbing, etc.)

2. **RL Enhancement** (inspired by Solving-CVRP-by-ALNS-and-RL)
   - Graph Neural Network (GIN) for solution encoding
   - Actor-Critic architecture for learned operator selection
   - Policy Gradient (REINFORCE) training
   - State-aware operator selection instead of adaptive weights

## 📁 Project Structure

```
AGV_ALNS_RL_Python/
├── models/
│   ├── problem/               # Problem definition
│   │   ├── task.py           # Task class
│   │   ├── agv.py            # AGV class
│   │   ├── charging_station.py
│   │   ├── solution.py       # Complete solution
│   │   └── schedule_builder.py
│   ├── alns/                  # ALNS framework
│   │   ├── base_alns.py      # Main ALNS algorithm
│   │   ├── acceptance.py     # Acceptance criteria
│   │   ├── weights.py        # Weight management
│   │   ├── initial_solution.py
│   │   └── operators/
│   │       ├── destroy.py    # 6 destroy operators
│   │       └── repair.py     # 9 repair operators
│   └── rl/                    # RL components
│       ├── gnn.py            # Graph Neural Network (GIN)
│       ├── actor.py          # Actor network
│       ├── critic.py         # Critic network
│       ├── graph_state.py    # Graph representation
│       └── rl_alns.py        # RL-enhanced ALNS
├── train.py                   # Training script for RL
├── solve.py                   # Main solving script
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
# Clone repository
cd AGV_ALNS_RL_Python

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start - Solve a Problem

```bash
# Compare both methods (base ALNS vs RL-ALNS)
python solve.py --method compare --problem small --iterations 5000

# Solve with base ALNS only
python solve.py --method base --problem small --iterations 5000

# Solve with RL-ALNS (untrained model)
python solve.py --method rl --problem small --iterations 5000

# Solve with pretrained RL model
python solve.py --method rl --problem small --model-path checkpoints/final_model.pt
```

### Generate Random Problems

```bash
# Solve random problem with 30 tasks, 4 AGVs, 3 stations
python solve.py --method compare --problem random \
    --num-tasks 30 --num-agvs 4 --num-stations 3 --iterations 5000
```

### Train RL Model

```bash
# Train with default settings
python train.py --epochs 200 --episodes 10 --steps 50

# Train with custom settings
python train.py --epochs 500 --episodes 20 --steps 100 \
    --lr 1e-4 --device cuda --seed 42
```

Training logs will be saved to `runs/` (for TensorBoard) and checkpoints to `checkpoints/`.

### Monitor Training

```bash
# View training progress in TensorBoard
tensorboard --logdir runs
```

## 🧩 Problem Definition

### AGV Charging Scheduling Problem

**Objective**: Minimize makespan (total completion time) while satisfying:
- Task time windows
- Battery State of Charge (SOC) constraints
- Charging station capacity limits
- Task precedence requirements

**Components**:
- **Tasks**: Location, time windows, duration, energy consumption
- **AGVs**: Battery capacity, charging requirements, speed
- **Charging Stations**: Location, capacity, charging rate

## 🔧 ALNS Operators

### Destroy Operators (6)

1. **RandomRemoval**: Random task removal
2. **ChargingCriticalRemoval**: Remove tasks with longest charging wait times
3. **ChargingWorstRemoval**: Remove tasks with worst charging efficiency
4. **StationCriticalRemoval**: Remove tasks from busiest charging station
5. **StationRandomRemoval**: Remove tasks from random station
6. **StationWorstRemoval**: Remove tasks from worst-performing station

### Repair Operators (9)

3×3 Strategy Matrix:

| Selection | Best (Quality) | Random (Diversity) | Time (Makespan) |
|-----------|----------------|--------------------|-----------------|
| **Greedy** | GreedyBestRepair | GreedyRandomRepair | GreedyTimeRepair |
| **Adaptive** | AdaptiveBestRepair | AdaptiveRandomRepair | AdaptiveTimeRepair |
| **Random** | RandomBestRepair | RandomRandomRepair | RandomTimeRepair |

## 🤖 RL Architecture

### Graph Neural Network (GIN)

- **Input**: Solution state as graph (tasks = nodes, sequences = edges)
- **Architecture**: 3 GIN layers with skip connections
- **Output**: Graph embedding for decision making

### Actor-Critic

- **Actor**: Selects destroy + repair operators (policy network)
- **Critic**: Estimates state value (value network)
- **Training**: REINFORCE with baseline

### Key Features

- **State-aware selection**: GNN captures solution structure
- **Faster convergence**: Learns which operators work in different states
- **Generalization**: Trained on one problem size, works on others

## 📊 Example Output

```
============================================================
SOLVING WITH BASE ALNS (Adaptive Weights)
============================================================

Initial solution cost: 1245.32
Initial makespan: 856.45
Initial charging time: 142.30

ALNS started with initial cost: 1245.32
Iter 100: Best=1156.78, Current=1189.34, NoImp=12, Time=5.2s
Iter 200: Best=1089.45, Current=1098.67, NoImp=34, Time=10.8s
...

Best solution cost: 987.56
Best makespan: 698.23
Best charging time: 98.45
Improvement: 257.76
Solve time: 45.32s
```

## 🔬 Experimental Results

### Base ALNS vs RL-ALNS

- **Base ALNS**: Uses adaptive weights (roulette wheel selection)
- **RL-ALNS (untrained)**: Random operator selection
- **RL-ALNS (trained)**: Learned policy for operator selection

Expected improvements with trained RL model:
- 10-20% better solution quality
- 30-50% faster convergence
- Better operator diversity

## 🛠️ Advanced Usage

### Python API

```python
from models.problem import AGVSolution, Task, AGV, ChargingStation
from models.alns.base_alns import ALNS, ALNSConfig
from models.rl import RLALNS, RLALNSConfig, ActorCritic

# Create problem
tasks = [...]
agvs = [...]
stations = [...]
solution = AGVSolution(agvs, tasks, stations)

# Solve with base ALNS
config = ALNSConfig(max_iterations=5000)
alns = ALNS(config=config)
best_solution = alns.solve(solution)

# Solve with RL-ALNS
actor_critic = ActorCritic(...)
rl_config = RLALNSConfig(use_rl=True)
rl_alns = RLALNS(config=rl_config, actor_critic=actor_critic)
best_solution = rl_alns.solve(solution)
```

### Custom Operators

```python
from models.alns.operators import DestroyOperator, RepairOperator

class MyCustomDestroy(DestroyOperator):
    def destroy(self, solution, num_remove):
        # Your custom logic
        return removed_tasks

# Register and use
DESTROY_OPERATORS['my_custom'] = MyCustomDestroy
```

## 📝 Citation

If you use this code, please cite:

```
Original AGV Charging Problem: [Your C++ paper]
ALNS Framework: alns-framework-for-evsp
RL Enhancement: Solving-CVRP-by-ALNS-and-RL
```

## 📄 License

[Your License]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[Your Contact Information]

## 🙏 Acknowledgments

- alns-framework-for-evsp for the ALNS framework structure
- Solving-CVRP-by-ALNS-and-RL for the RL integration approach
- Original C++ AGV charging problem formulation
