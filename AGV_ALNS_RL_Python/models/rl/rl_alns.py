"""
RL-enhanced ALNS for AGV charging scheduling

Uses learned policy (Actor-Critic) to select operators instead of adaptive weights
"""
import torch
import time
import random
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..problem import AGVSolution
from ..alns.operators import DESTROY_OPERATORS, REPAIR_OPERATORS
from ..alns.acceptance import get_acceptance_criterion
from .graph_state import GraphState
from .critic import ActorCritic


class RLALNSConfig:
    """Configuration for RL-enhanced ALNS"""

    def __init__(
        self,
        max_iterations: int = 5000,
        max_time: float = 300.0,
        max_iterations_no_improvement: int = 1000,
        min_remove_pct: float = 0.1,
        max_remove_pct: float = 0.4,
        acceptance_criterion: str = 'simulated_annealing',
        acceptance_params: Optional[Dict] = None,
        use_rl: bool = True,
        epsilon: float = 0.1,  # Exploration rate
        random_seed: Optional[int] = None
    ):
        """Initialize RL-ALNS configuration"""
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.max_iterations_no_improvement = max_iterations_no_improvement
        self.min_remove_pct = min_remove_pct
        self.max_remove_pct = max_remove_pct
        self.acceptance_criterion = acceptance_criterion
        self.acceptance_params = acceptance_params or {}
        self.use_rl = use_rl
        self.epsilon = epsilon
        self.random_seed = random_seed


class RLALNS:
    """
    RL-enhanced Adaptive Large Neighborhood Search

    Uses Actor-Critic network to learn which operators work best
    in different solution states
    """

    def __init__(
        self,
        config: Optional[RLALNSConfig] = None,
        actor_critic: Optional[ActorCritic] = None,
        device: str = 'cpu'
    ):
        """
        Initialize RL-ALNS

        Args:
            config: RL-ALNS configuration
            actor_critic: Pre-trained Actor-Critic model (optional)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.config = config or RLALNSConfig()
        self.device = torch.device(device)

        # Set random seed
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)

        # Initialize operators
        self.destroy_operator_names = list(DESTROY_OPERATORS.keys())
        self.repair_operator_names = list(REPAIR_OPERATORS.keys())

        self.destroy_operators = {
            name: DESTROY_OPERATORS[name]() for name in self.destroy_operator_names
        }
        self.repair_operators = {
            name: REPAIR_OPERATORS[name]() for name in self.repair_operator_names
        }

        # Actor-Critic model
        if actor_critic is None and self.config.use_rl:
            # Create new model
            actor_critic = ActorCritic(
                node_input_dim=8,
                global_input_dim=4,
                hidden_dim=128,
                num_gnn_layers=3,
                num_destroy_operators=len(self.destroy_operator_names),
                num_repair_operators=len(self.repair_operator_names),
                dropout=0.1,
                shared_encoder=True
            )

        self.actor_critic = actor_critic
        if self.actor_critic is not None:
            self.actor_critic.to(self.device)
            self.actor_critic.eval()  # Set to evaluation mode

        # Acceptance criterion
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
        self.operator_history: List[Tuple[str, str]] = []
        self.operator_usage: Dict[str, int] = {
            name: 0 for name in self.destroy_operator_names + self.repair_operator_names
        }

    def select_operators_rl(self, solution: AGVSolution) -> Tuple[str, str]:
        """
        Select operators using RL policy

        Args:
            solution: Current solution

        Returns:
            (destroy_operator_name, repair_operator_name)
        """
        if self.actor_critic is None or random.random() < self.config.epsilon:
            # Epsilon-greedy: random selection
            destroy_name = random.choice(self.destroy_operator_names)
            repair_name = random.choice(self.repair_operator_names)
            return destroy_name, repair_name

        # Convert solution to graph
        graph_state = GraphState(solution)
        graph_state.to(self.device)

        node_features, edge_index, edge_features, global_features = graph_state.get_batch()

        # Get action from policy
        with torch.no_grad():
            destroy_idx, repair_idx, _, _, _ = self.actor_critic.get_action_and_value(
                node_features,
                edge_index,
                global_features,
                batch=None,
                deterministic=False  # Stochastic selection
            )

        destroy_name = self.destroy_operator_names[destroy_idx]
        repair_name = self.repair_operator_names[repair_idx]

        return destroy_name, repair_name

    def select_operators_random(self) -> Tuple[str, str]:
        """Random operator selection (fallback)"""
        destroy_name = random.choice(self.destroy_operator_names)
        repair_name = random.choice(self.repair_operator_names)
        return destroy_name, repair_name

    def solve(self, initial_solution: AGVSolution, verbose: bool = True) -> AGVSolution:
        """
        Run RL-ALNS to optimize solution

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
            print(f"RL-ALNS started with initial cost: {self.current_cost:.2f}")
            print(f"Using RL policy: {self.config.use_rl}")

        # Main loop
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
            print(f"\nRL-ALNS completed:")
            print(f"  Total iterations: {self.iteration}")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  Best cost: {self.best_cost:.2f}")
            print(f"  Improvement: {(initial_solution.calculate_objective() - self.best_cost):.2f}")
            self.print_operator_statistics()

        return self.best_solution

    def perform_iteration(self):
        """Perform one RL-ALNS iteration"""
        # Select operators
        if self.config.use_rl:
            destroy_name, repair_name = self.select_operators_rl(self.current_solution)
        else:
            destroy_name, repair_name = self.select_operators_random()

        # Update usage statistics
        self.operator_usage[destroy_name] += 1
        self.operator_usage[repair_name] += 1

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
        is_accepted = self.acceptance.accept(self.current_cost, new_cost, self.iteration)

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
        """Check stopping criteria"""
        if self.iteration >= self.config.max_iterations:
            return True

        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.config.max_time:
            return True

        if self.iterations_since_improvement >= self.config.max_iterations_no_improvement:
            return True

        return False

    def print_operator_statistics(self):
        """Print operator usage statistics"""
        print("\n=== Operator Usage Statistics ===")

        print("Destroy operators:")
        destroy_usage = {k: v for k, v in self.operator_usage.items()
                        if k in self.destroy_operator_names}
        for name, count in sorted(destroy_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = 100.0 * count / max(1, self.iteration)
            print(f"  {name:25s}: {count:4d} ({percentage:5.1f}%)")

        print("\nRepair operators:")
        repair_usage = {k: v for k, v in self.operator_usage.items()
                       if k in self.repair_operator_names}
        for name, count in sorted(repair_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = 100.0 * count / max(1, self.iteration)
            print(f"  {name:25s}: {count:4d} ({percentage:5.1f}%)")

    def get_solution_history(self) -> Dict:
        """Get solution history"""
        return {
            'cost_history': self.cost_history,
            'best_cost_history': self.best_cost_history,
            'operator_history': self.operator_history,
            'operator_usage': self.operator_usage
        }
