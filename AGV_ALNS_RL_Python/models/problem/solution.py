"""
Solution representation for AGV charging scheduling problem
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import copy
from .agv import AGV, EventType, ScheduleEvent
from .task import Task
from .charging_station import ChargingStation


class AGVSolution:
    """
    Represents a complete solution to the AGV charging scheduling problem

    A solution consists of:
    - Task assignments to AGVs
    - Schedules for each AGV (including tasks, travel, charging, idle)
    - Charging decisions (when and where to charge)
    """

    def __init__(
        self,
        agvs: List[AGV],
        tasks: List[Task],
        charging_stations: List[ChargingStation]
    ):
        """
        Initialize solution

        Args:
            agvs: List of available AGVs
            tasks: List of tasks to be completed
            charging_stations: List of available charging stations
        """
        self.agvs = agvs
        self.tasks = tasks
        self.charging_stations = charging_stations

        # Create lookup dictionaries for fast access
        self.task_dict: Dict[int, Task] = {task.id: task for task in tasks}
        self.station_dict: Dict[int, ChargingStation] = {
            station.id: station for station in charging_stations
        }
        self.agv_dict: Dict[int, AGV] = {agv.id: agv for agv in agvs}

        # Solution metrics
        self.makespan: float = 0.0
        self.total_charging_time: float = 0.0
        self.total_idle_time: float = 0.0
        self.total_energy_cost: float = 0.0
        self.is_feasible: bool = True
        self.violations: List[str] = []

        # Task assignment tracking
        self.task_to_agv: Dict[int, int] = {}  # task_id -> agv_id
        self.unassigned_tasks: List[int] = [task.id for task in tasks]

    def copy(self) -> 'AGVSolution':
        """Create a deep copy of this solution"""
        # Deep copy AGVs
        new_agvs = []
        for agv in self.agvs:
            new_agv = AGV(
                id=agv.id,
                battery_capacity=agv.battery_capacity,
                initial_soc=agv.initial_soc,
                min_soc=agv.min_soc,
                max_soc=agv.max_soc,
                speed=agv.speed,
                idle_consumption=agv.idle_consumption,
                travel_consumption=agv.travel_consumption,
                initial_location=agv.initial_location
            )
            new_agv.task_sequence = agv.task_sequence.copy()
            new_agv.schedule_events = copy.deepcopy(agv.schedule_events)
            new_agv.current_location = agv.current_location
            new_agv.current_time = agv.current_time
            new_agv.current_soc = agv.current_soc
            new_agvs.append(new_agv)

        # Copy charging stations (reset events)
        new_stations = []
        for station in self.charging_stations:
            new_station = ChargingStation(
                id=station.id,
                location=station.location,
                capacity=station.capacity,
                charging_rate=station.charging_rate,
                max_charging_time=station.max_charging_time
            )
            new_station.events = copy.deepcopy(station.events)
            new_stations.append(new_station)

        # Create new solution
        new_solution = AGVSolution(new_agvs, self.tasks, new_stations)
        new_solution.task_to_agv = self.task_to_agv.copy()
        new_solution.unassigned_tasks = self.unassigned_tasks.copy()
        new_solution.makespan = self.makespan
        new_solution.total_charging_time = self.total_charging_time
        new_solution.total_idle_time = self.total_idle_time
        new_solution.is_feasible = self.is_feasible
        new_solution.violations = self.violations.copy()

        return new_solution

    def assign_task_to_agv(self, task_id: int, agv_id: int):
        """Assign a task to an AGV"""
        if task_id in self.task_to_agv:
            # Remove from old AGV
            old_agv_id = self.task_to_agv[task_id]
            self.agv_dict[old_agv_id].remove_task(task_id)

        # Assign to new AGV
        self.agv_dict[agv_id].add_task(task_id)
        self.task_to_agv[task_id] = agv_id

        if task_id in self.unassigned_tasks:
            self.unassigned_tasks.remove(task_id)

    def remove_task_from_solution(self, task_id: int):
        """Remove a task from its assigned AGV"""
        if task_id in self.task_to_agv:
            agv_id = self.task_to_agv[task_id]
            self.agv_dict[agv_id].remove_task(task_id)
            del self.task_to_agv[task_id]
            if task_id not in self.unassigned_tasks:
                self.unassigned_tasks.append(task_id)

    def calculate_objective(
        self,
        makespan_weight: float = 1.0,
        charging_time_weight: float = 0.1,
        idle_time_weight: float = 0.1,
        energy_cost_weight: float = 0.01,
        violation_penalty: float = 10000.0
    ) -> float:
        """
        Calculate weighted objective value

        Args:
            makespan_weight: Weight for makespan minimization
            charging_time_weight: Weight for total charging time
            idle_time_weight: Weight for total idle time
            energy_cost_weight: Weight for energy cost
            violation_penalty: Penalty for constraint violations

        Returns:
            Objective value (lower is better)
        """
        objective = 0.0

        # Makespan (max completion time across all AGVs)
        objective += makespan_weight * self.makespan

        # Charging time
        objective += charging_time_weight * self.total_charging_time

        # Idle time
        objective += idle_time_weight * self.total_idle_time

        # Energy cost
        objective += energy_cost_weight * self.total_energy_cost

        # Violation penalties
        if not self.is_feasible:
            objective += violation_penalty * len(self.violations)

        # Penalty for unassigned tasks
        objective += violation_penalty * len(self.unassigned_tasks)

        return objective

    def update_metrics(self):
        """Recalculate all solution metrics"""
        # Reset metrics
        self.makespan = 0.0
        self.total_charging_time = 0.0
        self.total_idle_time = 0.0
        self.total_energy_cost = 0.0
        self.is_feasible = True
        self.violations = []

        # Aggregate from all AGVs
        for agv in self.agvs:
            agv_makespan = agv.get_makespan()
            self.makespan = max(self.makespan, agv_makespan)
            self.total_charging_time += agv.get_total_charging_time()
            self.total_idle_time += agv.get_total_idle_time()

            # Validate each AGV's schedule
            is_valid, agv_violations = agv.validate_schedule()
            if not is_valid:
                self.is_feasible = False
                self.violations.extend([f"AGV{agv.id}: {v}" for v in agv_violations])

        # Calculate energy cost
        for station in self.charging_stations:
            for event in station.events:
                self.total_energy_cost += event.energy_charged  # Can add price factor here

        # Check charging station capacity violations
        for station in self.charging_stations:
            peak_usage = station.get_peak_usage()
            if peak_usage > station.capacity:
                self.is_feasible = False
                self.violations.append(
                    f"Station {station.id}: Peak usage ({peak_usage}) > capacity ({station.capacity})"
                )

    def get_task_assignment_info(self) -> Dict[int, List[int]]:
        """
        Get task assignments organized by AGV

        Returns:
            Dictionary mapping agv_id to list of assigned task_ids
        """
        assignment_info = {agv.id: [] for agv in self.agvs}
        for task_id, agv_id in self.task_to_agv.items():
            assignment_info[agv_id].append(task_id)
        return assignment_info

    def get_solution_summary(self) -> Dict:
        """Get comprehensive solution summary"""
        return {
            'makespan': self.makespan,
            'total_charging_time': self.total_charging_time,
            'total_idle_time': self.total_idle_time,
            'total_energy_cost': self.total_energy_cost,
            'num_tasks': len(self.tasks),
            'num_assigned_tasks': len(self.task_to_agv),
            'num_unassigned_tasks': len(self.unassigned_tasks),
            'is_feasible': self.is_feasible,
            'num_violations': len(self.violations),
            'num_agvs_used': sum(1 for agv in self.agvs if len(agv.task_sequence) > 0),
            'agv_summaries': [agv.get_summary() for agv in self.agvs],
            'station_utilizations': [
                {
                    'id': station.id,
                    'utilization': station.get_utilization(self.makespan),
                    'peak_usage': station.get_peak_usage(),
                    'num_events': len(station.events)
                }
                for station in self.charging_stations
            ]
        }

    def __repr__(self) -> str:
        return (
            f"AGVSolution(makespan={self.makespan:.2f}, "
            f"tasks={len(self.task_to_agv)}/{len(self.tasks)}, "
            f"feasible={self.is_feasible})"
        )
