"""
Destroy operators for AGV charging scheduling ALNS

These operators remove tasks/charging decisions from the current solution
to create room for improvement through repair operators.
"""
import random
import numpy as np
from typing import List, Set
from ...problem import AGVSolution, AGV, EventType


class DestroyOperator:
    """Base class for destroy operators"""

    def __init__(self, name: str):
        self.name = name
        self.num_calls = 0
        self.total_score = 0.0

    def __call__(self, solution: AGVSolution, num_remove: int) -> List[int]:
        """
        Remove tasks from solution

        Args:
            solution: Current solution
            num_remove: Number of tasks to remove

        Returns:
            List of removed task IDs
        """
        self.num_calls += 1
        return self.destroy(solution, num_remove)

    def destroy(self, solution: AGVSolution, num_remove: int) -> List[int]:
        """
        Implement destroy logic (to be overridden)

        Args:
            solution: Current solution
            num_remove: Number of tasks to remove

        Returns:
            List of removed task IDs
        """
        raise NotImplementedError

    def get_avg_score(self) -> float:
        """Get average score for this operator"""
        return self.total_score / max(1, self.num_calls)


class RandomRemoval(DestroyOperator):
    """Remove random tasks from the solution"""

    def __init__(self):
        super().__init__("RandomRemoval")

    def destroy(self, solution: AGVSolution, num_remove: int) -> List[int]:
        """Randomly select and remove tasks"""
        # Get all assigned tasks
        assigned_tasks = list(solution.task_to_agv.keys())

        if not assigned_tasks:
            return []

        # Randomly select tasks to remove
        num_to_remove = min(num_remove, len(assigned_tasks))
        tasks_to_remove = random.sample(assigned_tasks, num_to_remove)

        # Remove from solution
        for task_id in tasks_to_remove:
            solution.remove_task_from_solution(task_id)

        return tasks_to_remove


class ChargingCriticalRemoval(DestroyOperator):
    """
    Remove tasks that have the most critical charging requirements
    (longest waiting time for charging)
    """

    def __init__(self):
        super().__init__("ChargingCriticalRemoval")

    def destroy(self, solution: AGVSolution, num_remove: int) -> List[int]:
        """Remove tasks with longest charging waiting times"""
        task_criticality = []

        # Calculate criticality for each task based on charging impact
        for agv in solution.agvs:
            if not agv.schedule_events:
                continue

            # Find charging events and their associated waiting times
            for i, event in enumerate(agv.schedule_events):
                if event.event_type == EventType.CHARGING:
                    # Calculate waiting time before charging
                    waiting_time = 0.0
                    if i > 0:
                        prev_event = agv.schedule_events[i - 1]
                        waiting_time = event.start_time - prev_event.end_time

                    # Find the task before this charging event
                    task_before = None
                    for j in range(i - 1, -1, -1):
                        if agv.schedule_events[j].event_type == EventType.TASK:
                            task_before = agv.schedule_events[j].task_id
                            break

                    if task_before is not None:
                        task_criticality.append((task_before, waiting_time))

        # If no charging-related tasks, fall back to random
        if not task_criticality:
            return RandomRemoval().destroy(solution, num_remove)

        # Sort by criticality (descending)
        task_criticality.sort(key=lambda x: x[1], reverse=True)

        # Remove most critical tasks
        tasks_to_remove = []
        for task_id, _ in task_criticality[:num_remove]:
            if task_id in solution.task_to_agv:
                tasks_to_remove.append(task_id)
                solution.remove_task_from_solution(task_id)

        return tasks_to_remove


class ChargingWorstRemoval(DestroyOperator):
    """
    Remove tasks with worst charging efficiency
    (largest time interval between task end and next task start due to charging)
    """

    def __init__(self):
        super().__init__("ChargingWorstRemoval")

    def destroy(self, solution: AGVSolution, num_remove: int) -> List[int]:
        """Remove tasks with worst charging intervals"""
        task_inefficiency = []

        # Calculate inefficiency for each task
        for agv in solution.agvs:
            for i in range(len(agv.schedule_events) - 1):
                current = agv.schedule_events[i]
                next_event = agv.schedule_events[i + 1]

                if current.event_type == EventType.TASK:
                    # Calculate time gap to next task
                    time_gap = next_event.start_time - current.end_time

                    # If there's charging in between, this is inefficient
                    has_charging = False
                    for j in range(i + 1, len(agv.schedule_events)):
                        if agv.schedule_events[j].event_type == EventType.CHARGING:
                            has_charging = True
                            break
                        if agv.schedule_events[j].event_type == EventType.TASK:
                            break

                    if has_charging:
                        task_inefficiency.append((current.task_id, time_gap))

        # Fall back to random if no inefficient tasks
        if not task_inefficiency:
            return RandomRemoval().destroy(solution, num_remove)

        # Sort by inefficiency (descending)
        task_inefficiency.sort(key=lambda x: x[1], reverse=True)

        # Remove worst tasks
        tasks_to_remove = []
        for task_id, _ in task_inefficiency[:num_remove]:
            if task_id in solution.task_to_agv:
                tasks_to_remove.append(task_id)
                solution.remove_task_from_solution(task_id)

        return tasks_to_remove


class StationCriticalRemoval(DestroyOperator):
    """
    Remove all tasks from AGVs that use the most congested charging station
    """

    def __init__(self):
        super().__init__("StationCriticalRemoval")

    def destroy(self, solution: AGVSolution, num_remove: int) -> List[int]:
        """Remove tasks using the busiest charging station"""
        if not solution.charging_stations:
            return RandomRemoval().destroy(solution, num_remove)

        # Find the busiest station
        busiest_station = max(
            solution.charging_stations,
            key=lambda s: len(s.events)
        )

        if not busiest_station.events:
            return RandomRemoval().destroy(solution, num_remove)

        # Get AGVs that use this station
        agv_ids_using_station = set(event.agv_id for event in busiest_station.events)

        # Collect tasks from these AGVs
        tasks_to_remove = []
        for agv_id in agv_ids_using_station:
            agv = solution.agv_dict.get(agv_id)
            if agv:
                for task_id in agv.task_sequence[:num_remove]:
                    if task_id in solution.task_to_agv:
                        tasks_to_remove.append(task_id)
                        if len(tasks_to_remove) >= num_remove:
                            break
            if len(tasks_to_remove) >= num_remove:
                break

        # Remove tasks
        for task_id in tasks_to_remove:
            solution.remove_task_from_solution(task_id)

        return tasks_to_remove


class StationRandomRemoval(DestroyOperator):
    """Remove tasks from AGVs using a randomly selected charging station"""

    def __init__(self):
        super().__init__("StationRandomRemoval")

    def destroy(self, solution: AGVSolution, num_remove: int) -> List[int]:
        """Remove tasks from random station users"""
        # Filter stations that have events
        active_stations = [s for s in solution.charging_stations if s.events]

        if not active_stations:
            return RandomRemoval().destroy(solution, num_remove)

        # Randomly select a station
        selected_station = random.choice(active_stations)

        # Get AGVs using this station
        agv_ids_using_station = set(event.agv_id for event in selected_station.events)

        # Collect and remove tasks
        tasks_to_remove = []
        for agv_id in agv_ids_using_station:
            agv = solution.agv_dict.get(agv_id)
            if agv and agv.task_sequence:
                # Remove random tasks from this AGV
                tasks_from_agv = random.sample(
                    agv.task_sequence,
                    min(num_remove - len(tasks_to_remove), len(agv.task_sequence))
                )
                tasks_to_remove.extend(tasks_from_agv)

            if len(tasks_to_remove) >= num_remove:
                break

        # Remove tasks
        for task_id in tasks_to_remove[:num_remove]:
            if task_id in solution.task_to_agv:
                solution.remove_task_from_solution(task_id)

        return tasks_to_remove[:num_remove]


class StationWorstRemoval(DestroyOperator):
    """Remove tasks from AGVs using the worst-performing charging station"""

    def __init__(self):
        super().__init__("StationWorstRemoval")

    def destroy(self, solution: AGVSolution, num_remove: int) -> List[int]:
        """Remove tasks from worst-performing station"""
        if not solution.charging_stations or solution.makespan == 0:
            return RandomRemoval().destroy(solution, num_remove)

        # Calculate station performance (lower utilization = worse)
        station_performance = []
        for station in solution.charging_stations:
            if station.events:
                utilization = station.get_utilization(solution.makespan)
                peak_usage = station.get_peak_usage()

                # Worst stations: low utilization but high peak (inefficient usage)
                # or very high utilization (potential bottleneck)
                if peak_usage > 0:
                    efficiency_score = utilization / peak_usage
                else:
                    efficiency_score = 0.0

                station_performance.append((station, efficiency_score))

        if not station_performance:
            return RandomRemoval().destroy(solution, num_remove)

        # Sort by performance (ascending - worst first)
        station_performance.sort(key=lambda x: x[1])

        # Select worst station
        worst_station = station_performance[0][0]

        # Get tasks from AGVs using this station
        agv_ids_using_station = set(event.agv_id for event in worst_station.events)

        tasks_to_remove = []
        for agv_id in agv_ids_using_station:
            agv = solution.agv_dict.get(agv_id)
            if agv:
                for task_id in agv.task_sequence:
                    if task_id in solution.task_to_agv:
                        tasks_to_remove.append(task_id)
                        if len(tasks_to_remove) >= num_remove:
                            break
            if len(tasks_to_remove) >= num_remove:
                break

        # Remove tasks
        for task_id in tasks_to_remove:
            solution.remove_task_from_solution(task_id)

        return tasks_to_remove


# Registry of all destroy operators
DESTROY_OPERATORS = {
    'random': RandomRemoval,
    'charging_critical': ChargingCriticalRemoval,
    'charging_worst': ChargingWorstRemoval,
    'station_critical': StationCriticalRemoval,
    'station_random': StationRandomRemoval,
    'station_worst': StationWorstRemoval
}


def get_destroy_operator(name: str) -> DestroyOperator:
    """Get destroy operator by name"""
    if name not in DESTROY_OPERATORS:
        raise ValueError(f"Unknown destroy operator: {name}")
    return DESTROY_OPERATORS[name]()
