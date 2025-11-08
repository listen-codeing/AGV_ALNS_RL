"""
Task definition for AGV charging scheduling problem
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Task:
    """
    Represents a task that needs to be executed by an AGV

    Attributes:
        id: Unique task identifier
        location: (x, y) coordinates of task location
        earliest_start: Earliest time the task can start
        latest_start: Latest time the task can start (time window)
        duration: Time required to complete the task
        energy_consumption: Energy consumed during task execution
        priority: Task priority (higher = more important)
    """
    id: int
    location: Tuple[float, float]
    earliest_start: float
    latest_start: float
    duration: float
    energy_consumption: float
    priority: int = 1

    def __post_init__(self):
        """Validate task parameters"""
        assert self.earliest_start <= self.latest_start, \
            f"Task {self.id}: earliest_start must be <= latest_start"
        assert self.duration > 0, f"Task {self.id}: duration must be positive"
        assert self.energy_consumption >= 0, f"Task {self.id}: energy consumption must be non-negative"

    def get_time_window(self) -> Tuple[float, float]:
        """Return the time window for this task"""
        return (self.earliest_start, self.latest_start)

    def get_completion_time(self, start_time: float) -> float:
        """Calculate task completion time given a start time"""
        return start_time + self.duration

    def is_feasible_start(self, start_time: float) -> bool:
        """Check if a given start time is within the time window"""
        return self.earliest_start <= start_time <= self.latest_start

    def __repr__(self) -> str:
        return f"Task({self.id}, loc={self.location}, tw=[{self.earliest_start:.1f}, {self.latest_start:.1f}])"
