"""
Initial solution generators for AGV charging scheduling problem
"""
import random
import numpy as np
from typing import List
from ..problem import AGVSolution, Task, AGV
from ..problem.schedule_builder import ScheduleBuilder


class InitialSolutionGenerator:
    """Generates initial feasible solutions for the AGV charging problem"""

    @staticmethod
    def greedy_nearest_task(solution: AGVSolution) -> AGVSolution:
        """
        Generate initial solution using greedy nearest task assignment

        Strategy:
        - For each task (sorted by earliest start time):
          - Assign to the AGV that can complete it earliest
          - Consider travel time, current location, and time windows

        Args:
            solution: Empty solution with AGVs, tasks, and stations initialized

        Returns:
            Solution with initial task assignments
        """
        # Sort tasks by earliest start time
        sorted_tasks = sorted(solution.tasks, key=lambda t: t.earliest_start)

        # Track AGV availability
        agv_available_time = {agv.id: 0.0 for agv in solution.agvs}
        agv_location = {agv.id: agv.initial_location for agv in solution.agvs}

        # Assign each task to best AGV
        for task in sorted_tasks:
            best_agv = None
            best_completion_time = float('inf')

            for agv in solution.agvs:
                # Calculate when this AGV could start the task
                travel_time = agv.get_travel_time(agv_location[agv.id], task.location)
                earliest_arrival = agv_available_time[agv.id] + travel_time
                task_start = max(earliest_arrival, task.earliest_start)

                # Check time window feasibility
                if task_start > task.latest_start:
                    continue

                task_completion = task_start + task.duration

                # Select AGV with earliest completion
                if task_completion < best_completion_time:
                    best_completion_time = task_completion
                    best_agv = agv

            # Assign task to best AGV (or first AGV if none feasible)
            if best_agv is None:
                best_agv = solution.agvs[0]  # Fallback

            solution.assign_task_to_agv(task.id, best_agv.id)

            # Update AGV state
            travel_time = best_agv.get_travel_time(agv_location[best_agv.id], task.location)
            arrival_time = agv_available_time[best_agv.id] + travel_time
            task_start = max(arrival_time, task.earliest_start)
            task_end = task_start + task.duration

            agv_available_time[best_agv.id] = task_end
            agv_location[best_agv.id] = task.location

        # Build complete schedules with charging
        builder = ScheduleBuilder(solution)
        builder.rebuild_all_schedules(insert_charging=True)

        return solution

    @staticmethod
    def greedy_earliest_deadline(solution: AGVSolution) -> AGVSolution:
        """
        Generate initial solution using earliest deadline first (EDF)

        Strategy:
        - Sort tasks by latest start time (deadline)
        - Assign each task to the AGV with minimum makespan increase

        Args:
            solution: Empty solution

        Returns:
            Solution with initial task assignments
        """
        # Sort tasks by deadline (latest start)
        sorted_tasks = sorted(solution.tasks, key=lambda t: t.latest_start)

        # Track AGV states
        agv_available_time = {agv.id: 0.0 for agv in solution.agvs}
        agv_location = {agv.id: agv.initial_location for agv in solution.agvs}

        for task in sorted_tasks:
            best_agv = None
            best_makespan_increase = float('inf')

            for agv in solution.agvs:
                current_makespan = agv_available_time[agv.id]

                # Calculate makespan increase if this task is added
                travel_time = agv.get_travel_time(agv_location[agv.id], task.location)
                arrival_time = current_makespan + travel_time
                task_start = max(arrival_time, task.earliest_start)

                if task_start > task.latest_start:
                    continue

                new_makespan = task_start + task.duration
                makespan_increase = new_makespan - current_makespan

                if makespan_increase < best_makespan_increase:
                    best_makespan_increase = makespan_increase
                    best_agv = agv

            if best_agv is None:
                best_agv = solution.agvs[0]

            solution.assign_task_to_agv(task.id, best_agv.id)

            # Update state
            travel_time = best_agv.get_travel_time(agv_location[best_agv.id], task.location)
            arrival_time = agv_available_time[best_agv.id] + travel_time
            task_start = max(arrival_time, task.earliest_start)
            task_end = task_start + task.duration

            agv_available_time[best_agv.id] = task_end
            agv_location[best_agv.id] = task.location

        # Build schedules
        builder = ScheduleBuilder(solution)
        builder.rebuild_all_schedules(insert_charging=True)

        return solution

    @staticmethod
    def random_assignment(solution: AGVSolution) -> AGVSolution:
        """
        Generate initial solution with random task assignment

        Args:
            solution: Empty solution

        Returns:
            Solution with random task assignments
        """
        tasks = solution.tasks.copy()
        random.shuffle(tasks)

        # Round-robin assignment
        for i, task in enumerate(tasks):
            agv = solution.agvs[i % len(solution.agvs)]
            solution.assign_task_to_agv(task.id, agv.id)

        # Build schedules
        builder = ScheduleBuilder(solution)
        builder.rebuild_all_schedules(insert_charging=True)

        return solution

    @staticmethod
    def balanced_workload(solution: AGVSolution) -> AGVSolution:
        """
        Generate initial solution with balanced workload across AGVs

        Strategy:
        - Sort tasks by processing time (duration + travel)
        - Assign each task to the AGV with least total workload

        Args:
            solution: Empty solution

        Returns:
            Solution with balanced task assignments
        """
        # Calculate task workload (duration + average travel)
        depot_location = solution.agvs[0].initial_location
        task_workloads = []

        for task in solution.tasks:
            avg_travel = np.sqrt(
                (task.location[0] - depot_location[0])**2 +
                (task.location[1] - depot_location[1])**2
            ) / solution.agvs[0].speed

            workload = task.duration + avg_travel
            task_workloads.append((task, workload))

        # Sort by workload (descending)
        task_workloads.sort(key=lambda x: x[1], reverse=True)

        # Track AGV workloads
        agv_workloads = {agv.id: 0.0 for agv in solution.agvs}

        # Assign to least loaded AGV
        for task, workload in task_workloads:
            # Find AGV with minimum workload
            min_agv_id = min(agv_workloads.keys(), key=lambda k: agv_workloads[k])
            solution.assign_task_to_agv(task.id, min_agv_id)
            agv_workloads[min_agv_id] += workload

        # Build schedules
        builder = ScheduleBuilder(solution)
        builder.rebuild_all_schedules(insert_charging=True)

        return solution

    @staticmethod
    def generate(solution: AGVSolution, method: str = 'greedy_nearest') -> AGVSolution:
        """
        Generate initial solution using specified method

        Args:
            solution: Empty solution
            method: One of ['greedy_nearest', 'greedy_edf', 'random', 'balanced']

        Returns:
            Solution with initial assignments
        """
        if method == 'greedy_nearest':
            return InitialSolutionGenerator.greedy_nearest_task(solution)
        elif method == 'greedy_edf':
            return InitialSolutionGenerator.greedy_earliest_deadline(solution)
        elif method == 'random':
            return InitialSolutionGenerator.random_assignment(solution)
        elif method == 'balanced':
            return InitialSolutionGenerator.balanced_workload(solution)
        else:
            raise ValueError(f"Unknown initial solution method: {method}")
