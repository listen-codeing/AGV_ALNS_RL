"""
Weight management for ALNS operators

Implements adaptive weight adjustment and roulette wheel selection
"""
import random
import numpy as np
from typing import List, Dict, Tuple


class WeightManager:
    """
    Manages operator weights and selection for ALNS

    Uses roulette wheel selection and adaptive weight updates
    based on operator performance
    """

    def __init__(
        self,
        operator_names: List[str],
        initial_weight: float = 1.0,
        reaction_factor: float = 0.5,
        sigma1: float = 33.0,  # Score for new best solution
        sigma2: float = 9.0,   # Score for better solution
        sigma3: float = 1.0,   # Score for accepted solution
        segment_length: int = 100
    ):
        """
        Initialize weight manager

        Args:
            operator_names: List of operator names
            initial_weight: Initial weight for all operators
            reaction_factor: How quickly weights adapt (0 = never, 1 = immediately)
            sigma1: Score for finding new global best
            sigma2: Score for finding better solution
            sigma3: Score for finding accepted solution
            segment_length: How often to update weights (in iterations)
        """
        self.operator_names = operator_names
        self.reaction_factor = reaction_factor
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.segment_length = segment_length

        # Initialize weights
        self.weights: Dict[str, float] = {
            name: initial_weight for name in operator_names
        }

        # Tracking for current segment
        self.segment_scores: Dict[str, float] = {name: 0.0 for name in operator_names}
        self.segment_calls: Dict[str, int] = {name: 0 for name in operator_names}
        self.iterations_in_segment = 0

        # Historical tracking
        self.total_calls: Dict[str, int] = {name: 0 for name in operator_names}
        self.total_scores: Dict[str, float] = {name: 0.0 for name in operator_names}
        self.weight_history: List[Dict[str, float]] = []

    def select_operator(self) -> str:
        """
        Select an operator using roulette wheel selection

        Returns:
            Selected operator name
        """
        # Get weights as array
        names = list(self.weights.keys())
        weights = np.array([self.weights[name] for name in names])

        # Normalize to probabilities
        total_weight = weights.sum()
        if total_weight <= 0:
            # Fallback to uniform selection
            return random.choice(names)

        probabilities = weights / total_weight

        # Roulette wheel selection
        selected_name = np.random.choice(names, p=probabilities)
        return selected_name

    def update_score(
        self,
        operator_name: str,
        is_new_best: bool,
        is_better: bool,
        is_accepted: bool
    ):
        """
        Update operator score based on performance

        Args:
            operator_name: Name of the operator
            is_new_best: Whether the solution is a new global best
            is_better: Whether the solution is better than current
            is_accepted: Whether the solution was accepted
        """
        if operator_name not in self.operator_names:
            return

        # Calculate score
        score = 0.0
        if is_new_best:
            score = self.sigma1
        elif is_better:
            score = self.sigma2
        elif is_accepted:
            score = self.sigma3

        # Update segment tracking
        self.segment_scores[operator_name] += score
        self.segment_calls[operator_name] += 1
        self.total_calls[operator_name] += 1
        self.total_scores[operator_name] += score

        # Increment iteration counter
        self.iterations_in_segment += 1

        # Update weights if segment is complete
        if self.iterations_in_segment >= self.segment_length:
            self.update_weights()

    def update_weights(self):
        """Update operator weights based on segment performance"""
        for name in self.operator_names:
            if self.segment_calls[name] > 0:
                # Average score in this segment
                avg_score = self.segment_scores[name] / self.segment_calls[name]

                # Adaptive update: w_new = w_old * (1 - r) + r * avg_score
                old_weight = self.weights[name]
                new_weight = old_weight * (1 - self.reaction_factor) + \
                            self.reaction_factor * avg_score

                self.weights[name] = max(0.01, new_weight)  # Ensure minimum weight

        # Save weight history
        self.weight_history.append(self.weights.copy())

        # Reset segment tracking
        self.segment_scores = {name: 0.0 for name in self.operator_names}
        self.segment_calls = {name: 0 for name in self.operator_names}
        self.iterations_in_segment = 0

    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all operators

        Returns:
            Dictionary with operator statistics
        """
        stats = {}
        for name in self.operator_names:
            stats[name] = {
                'current_weight': self.weights[name],
                'total_calls': self.total_calls[name],
                'total_score': self.total_scores[name],
                'avg_score': self.total_scores[name] / max(1, self.total_calls[name]),
                'success_rate': self.total_calls[name] / max(1, sum(self.total_calls.values()))
            }
        return stats

    def get_best_operators(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k operators by average score

        Args:
            top_k: Number of top operators to return

        Returns:
            List of (operator_name, avg_score) tuples
        """
        operator_scores = []
        for name in self.operator_names:
            if self.total_calls[name] > 0:
                avg_score = self.total_scores[name] / self.total_calls[name]
                operator_scores.append((name, avg_score))

        # Sort by score (descending)
        operator_scores.sort(key=lambda x: x[1], reverse=True)

        return operator_scores[:top_k]

    def reset_segment(self):
        """Reset current segment tracking"""
        self.segment_scores = {name: 0.0 for name in self.operator_names}
        self.segment_calls = {name: 0 for name in self.operator_names}
        self.iterations_in_segment = 0

    def reset_all(self, initial_weight: float = 1.0):
        """Reset all tracking and weights"""
        self.weights = {name: initial_weight for name in self.operator_names}
        self.segment_scores = {name: 0.0 for name in self.operator_names}
        self.segment_calls = {name: 0 for name in self.operator_names}
        self.total_calls = {name: 0 for name in self.operator_names}
        self.total_scores = {name: 0.0 for name in self.operator_names}
        self.iterations_in_segment = 0
        self.weight_history = []
