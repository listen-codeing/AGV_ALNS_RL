"""
AGV (Automated Guided Vehicle) definition for charging scheduling problem
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np


class EventType(Enum):
    """Types of events in AGV schedule"""
    TASK = "TASK"
    CHARGING = "CHARGING"
    TRAVEL = "TRAVEL"
    IDLE = "IDLE"


@dataclass
class ScheduleEvent:
    """Represents an event in AGV's schedule"""
    event_type: EventType
    start_time: float
    end_time: float
    location: Tuple[float, float]
    task_id: Optional[int] = None
    station_id: Optional[int] = None
    energy_change: float = 0.0  # Negative for consumption, positive for charging
    soc_before: float = 0.0
    soc_after: float = 0.0

    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        if self.event_type == EventType.TASK:
            return f"TASK{self.task_id}[{self.start_time:.1f}-{self.end_time:.1f}]"
        elif self.event_type == EventType.CHARGING:
            return f"CHARGE@CS{self.station_id}[{self.start_time:.1f}-{self.end_time:.1f}]"
        else:
            return f"{self.event_type.value}[{self.start_time:.1f}-{self.end_time:.1f}]"


class AGV:
    """
    Represents an Automated Guided Vehicle

    Attributes:
        id: Unique AGV identifier
        battery_capacity: Maximum battery capacity (e.g., kWh)
        initial_soc: Initial state of charge (0.0 to 1.0)
        min_soc: Minimum allowed SOC (safety threshold)
        max_soc: Maximum SOC (typically 1.0)
        speed: Travel speed of the AGV
        idle_consumption: Energy consumption per time unit when idle
        travel_consumption: Energy consumption per distance unit
    """

    def __init__(
        self,
        id: int,
        battery_capacity: float,
        initial_soc: float = 1.0,
        min_soc: float = 0.2,
        max_soc: float = 1.0,
        speed: float = 1.0,
        idle_consumption: float = 0.01,
        travel_consumption: float = 0.05,
        initial_location: Tuple[float, float] = (0.0, 0.0)
    ):
        self.id = id
        self.battery_capacity = battery_capacity
        self.initial_soc = initial_soc
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.speed = speed
        self.idle_consumption = idle_consumption
        self.travel_consumption = travel_consumption
        self.initial_location = initial_location

        # Schedule components
        self.task_sequence: List[int] = []  # List of task IDs
        self.schedule_events: List[ScheduleEvent] = []
        self.current_location: Tuple[float, float] = initial_location
        self.current_time: float = 0.0
        self.current_soc: float = initial_soc

    def reset(self):
        """Reset AGV to initial state"""
        self.task_sequence = []
        self.schedule_events = []
        self.current_location = self.initial_location
        self.current_time = 0.0
        self.current_soc = self.initial_soc

    def add_task(self, task_id: int):
        """Add a task to this AGV's sequence"""
        self.task_sequence.append(task_id)

    def remove_task(self, task_id: int) -> bool:
        """Remove a task from this AGV's sequence"""
        if task_id in self.task_sequence:
            self.task_sequence.remove(task_id)
            return True
        return False

    def get_travel_time(self, from_loc: Tuple[float, float], to_loc: Tuple[float, float]) -> float:
        """Calculate travel time between two locations"""
        distance = np.sqrt((to_loc[0] - from_loc[0])**2 + (to_loc[1] - from_loc[1])**2)
        return distance / self.speed if self.speed > 0 else 0.0

    def get_travel_energy(self, from_loc: Tuple[float, float], to_loc: Tuple[float, float]) -> float:
        """Calculate energy consumption for travel between two locations"""
        distance = np.sqrt((to_loc[0] - from_loc[0])**2 + (to_loc[1] - from_loc[1])**2)
        return distance * self.travel_consumption

    def calculate_charging_time(self, energy_needed: float, charging_rate: float) -> float:
        """
        Calculate time needed to charge a specific amount of energy

        Args:
            energy_needed: Amount of energy to charge (in absolute units, not SOC)
            charging_rate: Charging rate of the station

        Returns:
            Time required for charging
        """
        if charging_rate <= 0:
            return float('inf')
        return energy_needed / charging_rate

    def calculate_energy_to_soc(self, target_soc: float, current_soc: float) -> float:
        """
        Convert SOC difference to absolute energy amount

        Args:
            target_soc: Target state of charge (0.0 to 1.0)
            current_soc: Current state of charge (0.0 to 1.0)

        Returns:
            Energy amount in absolute units
        """
        return (target_soc - current_soc) * self.battery_capacity

    def is_energy_feasible(self, energy_required: float, current_soc: float) -> bool:
        """
        Check if AGV has enough energy for an operation

        Args:
            energy_required: Energy required for the operation
            current_soc: Current state of charge

        Returns:
            True if feasible, False otherwise
        """
        # Convert current SOC to absolute energy
        current_energy = current_soc * self.battery_capacity

        # Check if remaining energy after operation is above minimum
        remaining_energy = current_energy - energy_required
        min_energy = self.min_soc * self.battery_capacity

        return remaining_energy >= min_energy

    def get_soc_after_consumption(self, energy_consumed: float, current_soc: float) -> float:
        """Calculate SOC after consuming energy"""
        current_energy = current_soc * self.battery_capacity
        remaining_energy = current_energy - energy_consumed
        return max(0.0, remaining_energy / self.battery_capacity)

    def get_soc_after_charging(self, energy_charged: float, current_soc: float) -> float:
        """Calculate SOC after charging"""
        current_energy = current_soc * self.battery_capacity
        new_energy = current_energy + energy_charged
        return min(self.max_soc, new_energy / self.battery_capacity)

    def get_makespan(self) -> float:
        """Get total time to complete all tasks (makespan)"""
        if not self.schedule_events:
            return 0.0
        return max(event.end_time for event in self.schedule_events)

    def get_total_charging_time(self) -> float:
        """Calculate total time spent charging"""
        return sum(
            event.duration()
            for event in self.schedule_events
            if event.event_type == EventType.CHARGING
        )

    def get_total_idle_time(self) -> float:
        """Calculate total idle/waiting time"""
        return sum(
            event.duration()
            for event in self.schedule_events
            if event.event_type == EventType.IDLE
        )

    def get_num_charging_sessions(self) -> int:
        """Count number of charging sessions"""
        return sum(
            1 for event in self.schedule_events
            if event.event_type == EventType.CHARGING
        )

    def validate_schedule(self) -> Tuple[bool, List[str]]:
        """
        Validate the complete schedule for feasibility

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Check SOC constraints
        for i, event in enumerate(self.schedule_events):
            if event.soc_before < self.min_soc - 1e-6:
                violations.append(
                    f"Event {i} ({event.event_type.value}): SOC before ({event.soc_before:.3f}) "
                    f"< min_soc ({self.min_soc:.3f})"
                )
            if event.soc_after < self.min_soc - 1e-6:
                violations.append(
                    f"Event {i} ({event.event_type.value}): SOC after ({event.soc_after:.3f}) "
                    f"< min_soc ({self.min_soc:.3f})"
                )
            if event.soc_before > self.max_soc + 1e-6:
                violations.append(
                    f"Event {i} ({event.event_type.value}): SOC before ({event.soc_before:.3f}) "
                    f"> max_soc ({self.max_soc:.3f})"
                )

        # Check time consistency
        for i in range(len(self.schedule_events) - 1):
            current = self.schedule_events[i]
            next_event = self.schedule_events[i + 1]
            if current.end_time > next_event.start_time + 1e-6:
                violations.append(
                    f"Events {i}-{i+1}: Time overlap "
                    f"({current.end_time:.2f} > {next_event.start_time:.2f})"
                )

        return len(violations) == 0, violations

    def __repr__(self) -> str:
        return (
            f"AGV({self.id}, tasks={len(self.task_sequence)}, "
            f"events={len(self.schedule_events)}, SOC={self.current_soc:.2f})"
        )

    def get_summary(self) -> Dict:
        """Get summary statistics for this AGV"""
        return {
            'id': self.id,
            'num_tasks': len(self.task_sequence),
            'makespan': self.get_makespan(),
            'charging_time': self.get_total_charging_time(),
            'idle_time': self.get_total_idle_time(),
            'num_charges': self.get_num_charging_sessions(),
            'final_soc': self.schedule_events[-1].soc_after if self.schedule_events else self.initial_soc
        }
