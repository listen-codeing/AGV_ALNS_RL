"""
Base ALNS (Adaptive Large Neighborhood Search) framework

Main algorithm that coordinates destroy/repair operators, acceptance criterion,
and weight management.
"""
import time
import random
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..problem import AGVSolution
from .operators import DESTROY_OPERATORS, REPAIR_OPERATORS
from .acceptance import AcceptanceCriterion, get_acceptance_criterion
from .weights import WeightManager


class ALNSConfig:
    """Configuration parameters for ALNS"""

    def __init__(
        self,
        max_iterations: int = 10000,
        max_time: float = 600.0,  # seconds
        max_iterations_no_improvement: int = 2000,
        min_remove_pct: float = 0.1,
        max_remove_pct: float = 0.4,
        reaction_factor: float = 0.5,
        sigma1: float = 33.0,
        sigma2: float = 9.0,
        sigma3: float = 1.0,
        segment_length: int = 100,
        acceptance_criterion: str = 'simulated_annealing',
        acceptance_params: Optional[Dict] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize ALNS configuration

        Args:
            max_iterations: Maximum number of iterations
            max_time: Maximum running time in seconds
            max_iterations_no_improvement: Stop after this many iterations without improvement
            min_remove_pct: Minimum percentage of tasks to remove
            max_remove_pct: Maximum percentage of tasks to remove
            reaction_factor: Weight update reaction factor
            sigma1: Score for new best solution
            sigma2: Score for better solution
            sigma3: Score for accepted solution
            segment_length: Iterations per weight update
            acceptance_criterion: Type of acceptance criterion
            acceptance_params: Parameters for acceptance criterion
            random_seed: Random seed for reproducibility
        """
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.max_iterations_no_improvement = max_iterations_no_improvement
        self.min_remove_pct = min_remove_pct
        self.max_remove_pct = max_remove_pct
        self.reaction_factor = reaction_factor
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.segment_length = segment_length
        self.acceptance_criterion = acceptance_criterion
        self.acceptance_params = acceptance_params or {}
        self.random_seed = random_seed


class ALNS:
    """
    Adaptive Large Neighborhood Search algorithm for AGV charging scheduling
    """

    def __init__(
        self,
        config: Optional[ALNSConfig] = None,
        destroy_operators: Optional[List[str]] = None,
        repair_operators: Optional[List[str]] = None
    ):
        """
        Initialize ALNS

        Args:
            config: ALNS configuration
            destroy_operators: List of destroy operator names to use
            repair_operators: List of repair operator names to use
        """
        self.config = config or ALNSConfig()

        # Set random seed
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        # Initialize operators
        if destroy_operators is None:
            destroy_operators = list(DESTROY_OPERATORS.keys())
        if repair_operators is None:
            repair_operators = list(REPAIR_OPERATORS.keys())

        self.destroy_operator_names = destroy_operators
        self.repair_operator_names = repair_operators

        # Create operator instances
        self.destroy_operators = {
            name: DESTROY_OPERATORS[name]() for name in destroy_operators
        }
        self.repair_operators = {
            name: REPAIR_OPERATORS[name]() for name in repair_operators
        }

        # Initialize weight managers
        self.destroy_weights = WeightManager(
            operator_names=destroy_operators,
            reaction_factor=self.config.reaction_factor,
            sigma1=self.config.sigma1,
            sigma2=self.config.sigma2,
            sigma3=self.config.sigma3,
            segment_length=self.config.segment_length
        )

        self.repair_weights = WeightManager(
            operator_names=repair_operators,
            reaction_factor=self.config.reaction_factor,
            sigma1=self.config.sigma1,
            sigma2=self.config.sigma2,
            sigma3=self.config.sigma3,
            segment_length=self.config.segment_length
        )

        # Initialize acceptance criterion
        self.acceptance = get_acceptance_criterion(
            self.config.acceptance_criterion,
            **self.config.acceptance_params
        )

        # Tracking
        self.best_solution: Optional[AGVSolution] = None
        self.best_cost: float = float('inf')
        self.current_solution: Optional[AGVSolution] = None
        self.current_cost: float = float('inf')

        self.iteration = 0
        self.iterations_since_improvement = 0
        self.start_time: float = 0.0

        # History
        self.cost_history: List[float] = []
        self.best_cost_history: List[float] = []
        self.operator_history: List[Tuple[str, str]] = []  # (destroy, repair)

    def solve(self, initial_solution: AGVSolution, verbose: bool = True) -> AGVSolution:
        """
        Run ALNS to optimize the solution

        Args:
            initial_solution: Starting solution
            verbose: Whether to print progress

        Returns:
            Best solution found
        """
        self.start_time = time.time()
        self.iteration = 0
        self.iterations_since_improvement = 0

        # Initialize solutions
        self.current_solution = initial_solution.copy()
        self.current_solution.update_metrics()
        self.current_cost = self.current_solution.calculate_objective()

        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_cost

        if verbose:
            print(f"ALNS started with initial cost: {self.current_cost:.2f}")
            print(f"Initial solution: {self.current_solution}")

        # Main ALNS loop
        while not self.should_stop():
            self.iteration += 1

            # Perform one iteration
            self.perform_iteration()

            # Log progress
            if verbose and self.iteration % 100 == 0:
                elapsed_time = time.time() - self.start_time
                print(
                    f"Iter {self.iteration}: Best={self.best_cost:.2f}, "
                    f"Current={self.current_cost:.2f}, "
                    f"NoImp={self.iterations_since_improvement}, "
                    f"Time={elapsed_time:.1f}s"
                )

        # Final report
        if verbose:
            elapsed_time = time.time() - self.start_time
            print(f"\nALNS completed:")
            print(f"  Total iterations: {self.iteration}")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  Best cost: {self.best_cost:.2f}")
            print(f"  Final solution: {self.best_solution}")
            self.print_operator_statistics()

        return self.best_solution

    def perform_iteration(self):
        """Perform one ALNS iteration"""
        # Select operators
        destroy_name = self.destroy_weights.select_operator()
        repair_name = self.repair_weights.select_operator()

        destroy_op = self.destroy_operators[destroy_name]
        repair_op = self.repair_operators[repair_name]

        # Calculate number of tasks to remove
        num_tasks = len(self.current_solution.tasks)
        min_remove = max(1, int(num_tasks * self.config.min_remove_pct))
        max_remove = max(min_remove, int(num_tasks * self.config.max_remove_pct))
        num_remove = random.randint(min_remove, max_remove)

        # Create copy for modification
        new_solution = self.current_solution.copy()

        # Apply destroy operator
        removed_tasks = destroy_op(new_solution, num_remove)

        # Apply repair operator
        repair_op(new_solution, removed_tasks)

        # Evaluate new solution
        new_solution.update_metrics()
        new_cost = new_solution.calculate_objective()

        # Determine solution quality
        is_new_best = new_cost < self.best_cost
        is_better = new_cost < self.current_cost
        is_accepted = self.acceptance.accept(self.current_cost, new_cost, self.iteration)

        # Update weights
        self.destroy_weights.update_score(destroy_name, is_new_best, is_better, is_accepted)
        self.repair_weights.update_score(repair_name, is_new_best, is_better, is_accepted)

        # Update solutions
        if is_accepted:
            self.current_solution = new_solution
            self.current_cost = new_cost

            if is_new_best:
                self.best_solution = new_solution.copy()
                self.best_cost = new_cost
                self.iterations_since_improvement = 0
            else:
                self.iterations_since_improvement += 1
        else:
            self.iterations_since_improvement += 1

        # Record history
        self.cost_history.append(self.current_cost)
        self.best_cost_history.append(self.best_cost)
        self.operator_history.append((destroy_name, repair_name))

    def should_stop(self) -> bool:
        """Check if stopping criteria are met"""
        # Max iterations
        if self.iteration >= self.config.max_iterations:
            return True

        # Max time
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.config.max_time:
            return True

        # Max iterations without improvement
        if self.iterations_since_improvement >= self.config.max_iterations_no_improvement:
            return True

        return False

    def print_operator_statistics(self):
        """Print statistics about operator performance"""
        print("\n=== Destroy Operator Statistics ===")
        destroy_stats = self.destroy_weights.get_weight_statistics()
        for name, stats in sorted(destroy_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True):
            print(
                f"  {name:25s}: weight={stats['current_weight']:6.2f}, "
                f"calls={stats['total_calls']:4d}, avg_score={stats['avg_score']:6.2f}"
            )

        print("\n=== Repair Operator Statistics ===")
        repair_stats = self.repair_weights.get_weight_statistics()
        for name, stats in sorted(repair_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True):
            print(
                f"  {name:25s}: weight={stats['current_weight']:6.2f}, "
                f"calls={stats['total_calls']:4d}, avg_score={stats['avg_score']:6.2f}"
            )

        print("\n=== Best Operator Combinations ===")
        print("Top destroy operators:", self.destroy_weights.get_best_operators(3))
        print("Top repair operators:", self.repair_weights.get_best_operators(3))

    def get_solution_history(self) -> Dict:
        """Get complete solution history"""
        return {
            'cost_history': self.cost_history,
            'best_cost_history': self.best_cost_history,
            'operator_history': self.operator_history,
            'destroy_stats': self.destroy_weights.get_weight_statistics(),
            'repair_stats': self.repair_weights.get_weight_statistics()
        }
