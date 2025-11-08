"""
Acceptance criteria for ALNS

Determines whether to accept a new solution based on solution quality
"""
import math
import random


class AcceptanceCriterion:
    """Base class for acceptance criteria"""

    def __init__(self, name: str):
        self.name = name

    def accept(self, current_cost: float, new_cost: float, iteration: int) -> bool:
        """
        Decide whether to accept new solution

        Args:
            current_cost: Cost of current solution
            new_cost: Cost of new solution
            iteration: Current iteration number

        Returns:
            True if new solution should be accepted, False otherwise
        """
        raise NotImplementedError


class SimulatedAnnealing(AcceptanceCriterion):
    """
    Simulated Annealing acceptance criterion

    Accepts worse solutions with probability exp(-delta/temperature)
    Temperature decreases over iterations
    """

    def __init__(
        self,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.9997,
        min_temperature: float = 0.01
    ):
        """
        Initialize simulated annealing

        Args:
            initial_temperature: Starting temperature (T0)
            cooling_rate: Temperature decay factor (alpha), T = T0 * alpha^iteration
            min_temperature: Minimum temperature threshold
        """
        super().__init__("SimulatedAnnealing")
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.current_temperature = initial_temperature

    def get_temperature(self, iteration: int) -> float:
        """
        Calculate temperature at given iteration

        Args:
            iteration: Current iteration number

        Returns:
            Current temperature
        """
        temp = self.initial_temperature * (self.cooling_rate ** iteration)
        return max(temp, self.min_temperature)

    def accept(self, current_cost: float, new_cost: float, iteration: int) -> bool:
        """
        Accept new solution based on simulated annealing criterion

        Args:
            current_cost: Cost of current solution
            new_cost: Cost of new solution
            iteration: Current iteration number

        Returns:
            True if accepted, False otherwise
        """
        # Always accept better solutions
        if new_cost < current_cost:
            return True

        # Update temperature
        self.current_temperature = self.get_temperature(iteration)

        # Accept worse solution with probability
        delta = new_cost - current_cost

        if self.current_temperature > 0:
            acceptance_probability = math.exp(-delta / self.current_temperature)

            if random.random() < acceptance_probability:
                return True

        return False

    def reset(self):
        """Reset temperature to initial value"""
        self.current_temperature = self.initial_temperature


class GreatDeluge(AcceptanceCriterion):
    """
    Great Deluge acceptance criterion

    Accepts solutions below a threshold (water level)
    Water level decreases linearly over iterations
    """

    def __init__(
        self,
        initial_level_factor: float = 1.05,
        decay_parameter: float = 0.01
    ):
        """
        Initialize Great Deluge

        Args:
            initial_level_factor: Initial water level = best_cost * factor
            decay_parameter: How fast the water level decreases
        """
        super().__init__("GreatDeluge")
        self.initial_level_factor = initial_level_factor
        self.decay_parameter = decay_parameter
        self.water_level = None

    def accept(self, current_cost: float, new_cost: float, iteration: int) -> bool:
        """Accept if new cost is below water level"""
        if self.water_level is None:
            self.water_level = current_cost * self.initial_level_factor

        # Accept if below water level
        if new_cost <= self.water_level:
            # Lower water level
            self.water_level -= self.decay_parameter
            return True

        return False

    def reset(self, initial_cost: float):
        """Reset water level"""
        self.water_level = initial_cost * self.initial_level_factor


class HillClimbing(AcceptanceCriterion):
    """
    Hill Climbing acceptance criterion

    Only accepts improving solutions
    """

    def __init__(self):
        super().__init__("HillClimbing")

    def accept(self, current_cost: float, new_cost: float, iteration: int) -> bool:
        """Only accept if new solution is better"""
        return new_cost < current_cost


class RecordToRecord(AcceptanceCriterion):
    """
    Record-to-Record Travel acceptance criterion

    Accepts solutions within a threshold of the best found so far
    """

    def __init__(self, threshold: float = 0.05):
        """
        Initialize Record-to-Record

        Args:
            threshold: Acceptance threshold relative to best solution
        """
        super().__init__("RecordToRecord")
        self.threshold = threshold
        self.record = float('inf')

    def accept(self, current_cost: float, new_cost: float, iteration: int) -> bool:
        """Accept if within threshold of best solution"""
        # Update record if better
        if new_cost < self.record:
            self.record = new_cost

        # Accept if within threshold
        return new_cost <= self.record * (1 + self.threshold)

    def reset(self):
        """Reset record"""
        self.record = float('inf')


# Factory function
def get_acceptance_criterion(name: str, **kwargs) -> AcceptanceCriterion:
    """
    Get acceptance criterion by name

    Args:
        name: One of ['simulated_annealing', 'great_deluge', 'hill_climbing', 'record_to_record']
        **kwargs: Parameters for the criterion

    Returns:
        AcceptanceCriterion instance
    """
    if name == 'simulated_annealing':
        return SimulatedAnnealing(**kwargs)
    elif name == 'great_deluge':
        return GreatDeluge(**kwargs)
    elif name == 'hill_climbing':
        return HillClimbing()
    elif name == 'record_to_record':
        return RecordToRecord(**kwargs)
    else:
        raise ValueError(f"Unknown acceptance criterion: {name}")
