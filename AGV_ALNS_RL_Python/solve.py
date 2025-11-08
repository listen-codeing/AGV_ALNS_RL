"""
Main solving script for AGV charging scheduling

Demonstrates both base ALNS and RL-enhanced ALNS
"""
import argparse
import random
import numpy as np
import torch
import json
import time

from models.problem import AGVSolution, Task, AGV, ChargingStation
from models.alns.initial_solution import InitialSolutionGenerator
from models.alns.base_alns import ALNS, ALNSConfig
from models.rl import RLALNS, RLALNSConfig, ActorCritic


def create_example_problem_small():
    """Create a small example problem (20 tasks, 3 AGVs, 2 stations)"""

    # Tasks
    tasks = [
        Task(0, (10, 20), 0, 100, 20, 8),
        Task(1, (30, 40), 10, 120, 25, 10),
        Task(2, (50, 30), 20, 130, 15, 6),
        Task(3, (70, 60), 15, 140, 30, 12),
        Task(4, (20, 80), 25, 150, 20, 8),
        Task(5, (40, 10), 30, 160, 18, 7),
        Task(6, (60, 50), 35, 170, 22, 9),
        Task(7, (80, 20), 40, 180, 25, 10),
        Task(8, (15, 45), 45, 190, 20, 8),
        Task(9, (35, 70), 50, 200, 28, 11),
        Task(10, (55, 15), 55, 210, 15, 6),
        Task(11, (75, 45), 60, 220, 24, 10),
        Task(12, (25, 60), 65, 230, 20, 8),
        Task(13, (45, 25), 70, 240, 17, 7),
        Task(14, (65, 80), 75, 250, 22, 9),
        Task(15, (85, 35), 80, 260, 26, 10),
        Task(16, (30, 55), 85, 270, 19, 8),
        Task(17, (50, 65), 90, 280, 21, 9),
        Task(18, (70, 30), 95, 290, 23, 9),
        Task(19, (90, 75), 100, 300, 27, 11),
    ]

    # Charging stations
    stations = [
        ChargingStation(0, (50, 50), capacity=2, charging_rate=15.0),
        ChargingStation(1, (20, 20), capacity=2, charging_rate=12.0),
    ]

    # AGVs
    agvs = [
        AGV(0, battery_capacity=100.0, initial_soc=1.0, min_soc=0.2, speed=2.0),
        AGV(1, battery_capacity=100.0, initial_soc=1.0, min_soc=0.2, speed=2.0),
        AGV(2, battery_capacity=100.0, initial_soc=1.0, min_soc=0.2, speed=2.0),
    ]

    return AGVSolution(agvs, tasks, stations)


def create_random_problem(num_tasks=30, num_agvs=4, num_stations=3, seed=None):
    """Create a random problem instance"""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid_size = 100.0

    # Generate tasks
    tasks = []
    for i in range(num_tasks):
        location = (random.uniform(0, grid_size), random.uniform(0, grid_size))
        earliest_start = random.uniform(0, 500)
        latest_start = earliest_start + random.uniform(50, 200)
        duration = random.uniform(10, 50)
        energy_consumption = random.uniform(5, 20)

        tasks.append(Task(
            id=i,
            location=location,
            earliest_start=earliest_start,
            latest_start=latest_start,
            duration=duration,
            energy_consumption=energy_consumption
        ))

    # Generate stations
    stations = []
    for i in range(num_stations):
        location = (random.uniform(0, grid_size), random.uniform(0, grid_size))
        stations.append(ChargingStation(
            id=i,
            location=location,
            capacity=random.randint(2, 3),
            charging_rate=random.uniform(10.0, 15.0)
        ))

    # Generate AGVs
    agvs = []
    for i in range(num_agvs):
        agvs.append(AGV(
            id=i,
            battery_capacity=100.0,
            initial_soc=1.0,
            min_soc=0.2,
            speed=2.0
        ))

    return AGVSolution(agvs, tasks, stations)


def solve_with_base_alns(problem: AGVSolution, max_iterations=5000, verbose=True):
    """Solve using base ALNS with adaptive weights"""

    if verbose:
        print("\n" + "="*60)
        print("SOLVING WITH BASE ALNS (Adaptive Weights)")
        print("="*60)

    # Generate initial solution
    initial_solution = InitialSolutionGenerator.generate(problem, method='greedy_nearest')

    if verbose:
        print(f"\nInitial solution cost: {initial_solution.calculate_objective():.2f}")
        print(f"Initial makespan: {initial_solution.makespan:.2f}")
        print(f"Initial charging time: {initial_solution.total_charging_time:.2f}")

    # Configure ALNS
    config = ALNSConfig(
        max_iterations=max_iterations,
        max_time=300.0,
        max_iterations_no_improvement=1000,
        acceptance_criterion='simulated_annealing',
        acceptance_params={'initial_temperature': 100.0, 'cooling_rate': 0.9995}
    )

    # Solve
    alns = ALNS(config=config)
    start_time = time.time()
    best_solution = alns.solve(initial_solution, verbose=verbose)
    solve_time = time.time() - start_time

    if verbose:
        print(f"\n{'='*60}")
        print(f"Best solution cost: {best_solution.calculate_objective():.2f}")
        print(f"Best makespan: {best_solution.makespan:.2f}")
        print(f"Best charging time: {best_solution.total_charging_time:.2f}")
        print(f"Improvement: {initial_solution.calculate_objective() - best_solution.calculate_objective():.2f}")
        print(f"Solve time: {solve_time:.2f}s")
        print(f"{'='*60}\n")

    return best_solution, alns


def solve_with_rl_alns(
    problem: AGVSolution,
    model_path: str = None,
    max_iterations=5000,
    use_rl=True,
    verbose=True
):
    """Solve using RL-enhanced ALNS"""

    if verbose:
        print("\n" + "="*60)
        print(f"SOLVING WITH RL-ENHANCED ALNS (use_rl={use_rl})")
        print("="*60)

    # Generate initial solution
    initial_solution = InitialSolutionGenerator.generate(problem, method='greedy_nearest')

    if verbose:
        print(f"\nInitial solution cost: {initial_solution.calculate_objective():.2f}")
        print(f"Initial makespan: {initial_solution.makespan:.2f}")
        print(f"Initial charging time: {initial_solution.total_charging_time:.2f}")

    # Load or create model
    actor_critic = None
    if use_rl:
        actor_critic = ActorCritic(
            node_input_dim=8,
            global_input_dim=4,
            hidden_dim=128,
            num_gnn_layers=3,
            num_destroy_operators=6,
            num_repair_operators=9,
            dropout=0.1,
            shared_encoder=True
        )

        if model_path and os.path.exists(model_path):
            actor_critic.load_state_dict(torch.load(model_path, map_location='cpu'))
            if verbose:
                print(f"Loaded pretrained model from {model_path}")
        else:
            if verbose:
                print("Using untrained RL model (random policy)")

    # Configure RL-ALNS
    config = RLALNSConfig(
        max_iterations=max_iterations,
        max_time=300.0,
        max_iterations_no_improvement=1000,
        acceptance_criterion='simulated_annealing',
        acceptance_params={'initial_temperature': 100.0, 'cooling_rate': 0.9995},
        use_rl=use_rl,
        epsilon=0.1
    )

    # Solve
    rl_alns = RLALNS(config=config, actor_critic=actor_critic, device='cpu')
    start_time = time.time()
    best_solution = rl_alns.solve(initial_solution, verbose=verbose)
    solve_time = time.time() - start_time

    if verbose:
        print(f"\n{'='*60}")
        print(f"Best solution cost: {best_solution.calculate_objective():.2f}")
        print(f"Best makespan: {best_solution.makespan:.2f}")
        print(f"Best charging time: {best_solution.total_charging_time:.2f}")
        print(f"Improvement: {initial_solution.calculate_objective() - best_solution.calculate_objective():.2f}")
        print(f"Solve time: {solve_time:.2f}s")
        print(f"{'='*60}\n")

    return best_solution, rl_alns


def compare_methods(problem: AGVSolution, max_iterations=5000):
    """Compare base ALNS and RL-ALNS"""

    print("\n" + "="*70)
    print("COMPARING ALNS METHODS")
    print("="*70)

    # Base ALNS
    base_solution, base_alns = solve_with_base_alns(
        problem.copy(),
        max_iterations=max_iterations,
        verbose=True
    )

    # RL-ALNS (without trained model - random policy)
    rl_solution, rl_alns = solve_with_rl_alns(
        problem.copy(),
        model_path=None,
        max_iterations=max_iterations,
        use_rl=True,
        verbose=True
    )

    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"Base ALNS cost:   {base_solution.calculate_objective():.2f}")
    print(f"RL-ALNS cost:     {rl_solution.calculate_objective():.2f}")
    print(f"Difference:       {base_solution.calculate_objective() - rl_solution.calculate_objective():.2f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Solve AGV charging scheduling problem')
    parser.add_argument('--method', type=str, default='compare',
                       choices=['base', 'rl', 'compare'],
                       help='Solution method')
    parser.add_argument('--problem', type=str, default='small',
                       choices=['small', 'random'],
                       help='Problem instance')
    parser.add_argument('--num-tasks', type=int, default=30,
                       help='Number of tasks (for random problem)')
    parser.add_argument('--num-agvs', type=int, default=4,
                       help='Number of AGVs (for random problem)')
    parser.add_argument('--num-stations', type=int, default=3,
                       help='Number of charging stations (for random problem)')
    parser.add_argument('--iterations', type=int, default=5000,
                       help='Maximum iterations')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pretrained RL model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create problem
    if args.problem == 'small':
        problem = create_example_problem_small()
        print(f"Created small problem: 20 tasks, 3 AGVs, 2 stations")
    else:
        problem = create_random_problem(
            num_tasks=args.num_tasks,
            num_agvs=args.num_agvs,
            num_stations=args.num_stations,
            seed=args.seed
        )
        print(f"Created random problem: {args.num_tasks} tasks, {args.num_agvs} AGVs, {args.num_stations} stations")

    # Solve
    import os
    if args.method == 'base':
        solve_with_base_alns(problem, max_iterations=args.iterations)
    elif args.method == 'rl':
        solve_with_rl_alns(
            problem,
            model_path=args.model_path,
            max_iterations=args.iterations,
            use_rl=True
        )
    else:  # compare
        compare_methods(problem, max_iterations=args.iterations)


if __name__ == '__main__':
    main()
