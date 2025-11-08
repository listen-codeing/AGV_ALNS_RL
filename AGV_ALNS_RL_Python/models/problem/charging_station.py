"""
Charging station definition for AGV charging scheduling problem
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class ChargingEvent:
    """Represents a single charging event at a station"""
    def __init__(self, agv_id: int, start_time: float, end_time: float, energy_charged: float):
        self.agv_id = agv_id
        self.start_time = start_time
        self.end_time = end_time
        self.energy_charged = energy_charged

    def overlaps_with(self, other: 'ChargingEvent') -> bool:
        """Check if this event overlaps with another charging event"""
        return not (self.end_time <= other.start_time or self.start_time >= other.end_time)

    def __repr__(self) -> str:
        return f"ChargingEvent(AGV{self.agv_id}, [{self.start_time:.1f}, {self.end_time:.1f}])"


@dataclass
class ChargingStation:
    """
    Represents a charging station for AGVs

    Attributes:
        id: Unique station identifier
        location: (x, y) coordinates of the station
        capacity: Maximum number of AGVs that can charge simultaneously
        charging_rate: Energy charging rate per time unit (e.g., kW)
        max_charging_time: Maximum continuous charging time allowed
    """
    id: int
    location: Tuple[float, float]
    capacity: int
    charging_rate: float  # Energy per time unit
    max_charging_time: float = float('inf')
    events: List[ChargingEvent] = field(default_factory=list)

    def get_available_slots(self, start_time: float, end_time: float) -> int:
        """
        Calculate how many charging slots are available in the given time window

        Args:
            start_time: Start of the time window
            end_time: End of the time window

        Returns:
            Number of available slots (0 to capacity)
        """
        # Count overlapping events
        overlapping = 0
        for event in self.events:
            if not (event.end_time <= start_time or event.start_time >= end_time):
                overlapping += 1

        return max(0, self.capacity - overlapping)

    def is_available(self, start_time: float, end_time: float) -> bool:
        """Check if at least one slot is available in the time window"""
        return self.get_available_slots(start_time, end_time) > 0

    def add_charging_event(self, agv_id: int, start_time: float, end_time: float,
                          energy_charged: float) -> bool:
        """
        Add a charging event to this station

        Returns:
            True if successful, False if station is at capacity
        """
        if not self.is_available(start_time, end_time):
            return False

        event = ChargingEvent(agv_id, start_time, end_time, energy_charged)
        self.events.append(event)
        return True

    def remove_charging_event(self, agv_id: int, start_time: float) -> bool:
        """
        Remove a specific charging event from this station

        Returns:
            True if event was found and removed, False otherwise
        """
        for i, event in enumerate(self.events):
            if event.agv_id == agv_id and abs(event.start_time - start_time) < 1e-6:
                self.events.pop(i)
                return True
        return False

    def clear_events_for_agv(self, agv_id: int):
        """Remove all charging events for a specific AGV"""
        self.events = [e for e in self.events if e.agv_id != agv_id]

    def get_utilization(self, time_horizon: float) -> float:
        """
        Calculate station utilization rate

        Args:
            time_horizon: Total time period to consider

        Returns:
            Utilization rate (0.0 to 1.0)
        """
        if time_horizon <= 0:
            return 0.0

        total_charging_time = sum(e.end_time - e.start_time for e in self.events)
        max_possible_time = self.capacity * time_horizon

        return min(1.0, total_charging_time / max_possible_time) if max_possible_time > 0 else 0.0

    def get_peak_usage(self) -> int:
        """Calculate peak concurrent usage of this station"""
        if not self.events:
            return 0

        # Create time points
        time_points = []
        for event in self.events:
            time_points.append((event.start_time, 1))  # +1 for start
            time_points.append((event.end_time, -1))   # -1 for end

        # Sort by time
        time_points.sort()

        # Find peak
        current_usage = 0
        peak_usage = 0
        for _, delta in time_points:
            current_usage += delta
            peak_usage = max(peak_usage, current_usage)

        return peak_usage

    def __repr__(self) -> str:
        return f"ChargingStation({self.id}, loc={self.location}, cap={self.capacity}, events={len(self.events)})"
