"""
Repair operators for AGV charging scheduling ALNS

These operators reinsert removed tasks back into the solution.
Strategy: 3x3 matrix
- Selection: Greedy, Random, Adaptive
- Optimization: Best (quality), Random (diversity), Time (makespan)
"""
import random
import numpy as np
from typing import List, Tuple, Optional
from ...problem import AGVSolution, AGV, Task
from ...problem.schedule_builder import ScheduleBuilder


class RepairOperator:
    """Base class for repair operators"""

    def __init__(self, name: str):
        self.name = name
        self.num_calls = 0
        self.total_score = 0.0

    def __call__(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """
        Repair solution by reinserting removed tasks

        Args:
            solution: Current solution
            removed_tasks: List of task IDs to reinsert

        Returns:
            Number of successfully inserted tasks
        """
        self.num_calls += 1
        return self.repair(solution, removed_tasks)

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Implement repair logic (to be overridden)"""
        raise NotImplementedError

    def get_avg_score(self) -> float:
        """Get average score for this operator"""
        return self.total_score / max(1, self.num_calls)


class GreedyBestRepair(RepairOperator):
    """
    Greedy selection + Best quality optimization
    Insert each task at the position that minimizes objective function
    """

    def __init__(self):
        super().__init__("GreedyBestRepair")

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Greedily insert tasks at best positions"""
        inserted_count = 0
        builder = ScheduleBuilder(solution)

        # Sort tasks by earliest start time for greedy order
        tasks_to_insert = [
            solution.task_dict[tid] for tid in removed_tasks
            if tid in solution.task_dict
        ]
        tasks_to_insert.sort(key=lambda t: t.earliest_start)

        for task in tasks_to_insert:
            best_agv = None
            best_position = -1
            best_cost = float('inf')

            # Try inserting into each AGV
            for agv in solution.agvs:
                # Try all positions in task sequence
                for pos in range(len(agv.task_sequence) + 1):
                    # Temporarily insert
                    agv.task_sequence.insert(pos, task.id)

                    # Rebuild schedule and evaluate
                    is_feasible, _ = builder.build_complete_schedule(agv, insert_charging=True)

                    if is_feasible:
                        # Calculate cost (makespan + charging time)
                        cost = agv.get_makespan() + agv.get_total_charging_time()

                        if cost < best_cost:
                            best_cost = cost
                            best_agv = agv
                            best_position = pos

                    # Remove temporary insertion
                    agv.task_sequence.pop(pos)

            # Insert at best position
            if best_agv is not None:
                best_agv.task_sequence.insert(best_position, task.id)
                solution.task_to_agv[task.id] = best_agv.id
                if task.id in solution.unassigned_tasks:
                    solution.unassigned_tasks.remove(task.id)
                inserted_count += 1

        # Rebuild all schedules
        builder.rebuild_all_schedules(insert_charging=True)
        return inserted_count


class GreedyRandomRepair(RepairOperator):
    """
    Greedy selection + Random diversity
    Insert each task at a random feasible position
    """

    def __init__(self):
        super().__init__("GreedyRandomRepair")

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Insert tasks at random feasible positions"""
        inserted_count = 0
        builder = ScheduleBuilder(solution)

        tasks_to_insert = [
            solution.task_dict[tid] for tid in removed_tasks
            if tid in solution.task_dict
        ]
        tasks_to_insert.sort(key=lambda t: t.earliest_start)

        for task in tasks_to_insert:
            feasible_positions = []

            # Find all feasible positions
            for agv in solution.agvs:
                for pos in range(len(agv.task_sequence) + 1):
                    agv.task_sequence.insert(pos, task.id)
                    is_feasible, _ = builder.build_complete_schedule(agv, insert_charging=True)

                    if is_feasible:
                        feasible_positions.append((agv, pos))

                    agv.task_sequence.pop(pos)

            # Randomly select from feasible positions
            if feasible_positions:
                selected_agv, selected_pos = random.choice(feasible_positions)
                selected_agv.task_sequence.insert(selected_pos, task.id)
                solution.task_to_agv[task.id] = selected_agv.id
                if task.id in solution.unassigned_tasks:
                    solution.unassigned_tasks.remove(task.id)
                inserted_count += 1

        builder.rebuild_all_schedules(insert_charging=True)
        return inserted_count


class GreedyTimeRepair(RepairOperator):
    """
    Greedy selection + Time optimization
    Insert each task to minimize makespan
    """

    def __init__(self):
        super().__init__("GreedyTimeRepair")

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Insert tasks to minimize makespan"""
        inserted_count = 0
        builder = ScheduleBuilder(solution)

        tasks_to_insert = [
            solution.task_dict[tid] for tid in removed_tasks
            if tid in solution.task_dict
        ]
        tasks_to_insert.sort(key=lambda t: t.earliest_start)

        for task in tasks_to_insert:
            best_agv = None
            best_position = -1
            best_makespan = float('inf')

            for agv in solution.agvs:
                for pos in range(len(agv.task_sequence) + 1):
                    agv.task_sequence.insert(pos, task.id)
                    is_feasible, _ = builder.build_complete_schedule(agv, insert_charging=True)

                    if is_feasible:
                        makespan = agv.get_makespan()
                        if makespan < best_makespan:
                            best_makespan = makespan
                            best_agv = agv
                            best_position = pos

                    agv.task_sequence.pop(pos)

            if best_agv is not None:
                best_agv.task_sequence.insert(best_position, task.id)
                solution.task_to_agv[task.id] = best_agv.id
                if task.id in solution.unassigned_tasks:
                    solution.unassigned_tasks.remove(task.id)
                inserted_count += 1

        builder.rebuild_all_schedules(insert_charging=True)
        return inserted_count


class AdaptiveBestRepair(RepairOperator):
    """
    Adaptive selection + Best quality
    Adaptively choose insertion order based on solution state
    """

    def __init__(self):
        super().__init__("AdaptiveBestRepair")

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Adaptively insert tasks for best quality"""
        inserted_count = 0
        builder = ScheduleBuilder(solution)

        # Adaptive ordering: prioritize tasks that are harder to insert
        tasks_to_insert = [
            solution.task_dict[tid] for tid in removed_tasks
            if tid in solution.task_dict
        ]

        # Calculate task difficulty (tighter time windows = harder)
        task_difficulty = []
        for task in tasks_to_insert:
            time_window_width = task.latest_start - task.earliest_start
            difficulty = 1.0 / (time_window_width + 1.0)  # Smaller window = higher difficulty
            task_difficulty.append((task, difficulty))

        # Sort by difficulty (descending)
        task_difficulty.sort(key=lambda x: x[1], reverse=True)

        for task, _ in task_difficulty:
            best_agv = None
            best_position = -1
            best_cost = float('inf')

            for agv in solution.agvs:
                for pos in range(len(agv.task_sequence) + 1):
                    agv.task_sequence.insert(pos, task.id)
                    is_feasible, _ = builder.build_complete_schedule(agv, insert_charging=True)

                    if is_feasible:
                        cost = agv.get_makespan() + agv.get_total_charging_time()
                        if cost < best_cost:
                            best_cost = cost
                            best_agv = agv
                            best_position = pos

                    agv.task_sequence.pop(pos)

            if best_agv is not None:
                best_agv.task_sequence.insert(best_position, task.id)
                solution.task_to_agv[task.id] = best_agv.id
                if task.id in solution.unassigned_tasks:
                    solution.unassigned_tasks.remove(task.id)
                inserted_count += 1

        builder.rebuild_all_schedules(insert_charging=True)
        return inserted_count


class AdaptiveRandomRepair(RepairOperator):
    """
    Adaptive selection + Random diversity
    Adaptively order tasks, then randomly insert
    """

    def __init__(self):
        super().__init__("AdaptiveRandomRepair")

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Adaptively order, randomly insert"""
        inserted_count = 0
        builder = ScheduleBuilder(solution)

        tasks_to_insert = [
            solution.task_dict[tid] for tid in removed_tasks
            if tid in solution.task_dict
        ]

        # Adaptive ordering by energy consumption
        tasks_to_insert.sort(key=lambda t: t.energy_consumption, reverse=True)

        for task in tasks_to_insert:
            feasible_positions = []

            for agv in solution.agvs:
                for pos in range(len(agv.task_sequence) + 1):
                    agv.task_sequence.insert(pos, task.id)
                    is_feasible, _ = builder.build_complete_schedule(agv, insert_charging=True)

                    if is_feasible:
                        feasible_positions.append((agv, pos))

                    agv.task_sequence.pop(pos)

            if feasible_positions:
                selected_agv, selected_pos = random.choice(feasible_positions)
                selected_agv.task_sequence.insert(selected_pos, task.id)
                solution.task_to_agv[task.id] = selected_agv.id
                if task.id in solution.unassigned_tasks:
                    solution.unassigned_tasks.remove(task.id)
                inserted_count += 1

        builder.rebuild_all_schedules(insert_charging=True)
        return inserted_count


class AdaptiveTimeRepair(RepairOperator):
    """
    Adaptive selection + Time optimization
    Adaptively order tasks, minimize makespan
    """

    def __init__(self):
        super().__init__("AdaptiveTimeRepair")

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Adaptively insert to minimize makespan"""
        inserted_count = 0
        builder = ScheduleBuilder(solution)

        tasks_to_insert = [
            solution.task_dict[tid] for tid in removed_tasks
            if tid in solution.task_dict
        ]

        # Adaptive: prioritize by deadline urgency
        tasks_to_insert.sort(key=lambda t: t.latest_start)

        for task in tasks_to_insert:
            best_agv = None
            best_position = -1
            best_makespan = float('inf')

            for agv in solution.agvs:
                for pos in range(len(agv.task_sequence) + 1):
                    agv.task_sequence.insert(pos, task.id)
                    is_feasible, _ = builder.build_complete_schedule(agv, insert_charging=True)

                    if is_feasible:
                        makespan = agv.get_makespan()
                        if makespan < best_makespan:
                            best_makespan = makespan
                            best_agv = agv
                            best_position = pos

                    agv.task_sequence.pop(pos)

            if best_agv is not None:
                best_agv.task_sequence.insert(best_position, task.id)
                solution.task_to_agv[task.id] = best_agv.id
                if task.id in solution.unassigned_tasks:
                    solution.unassigned_tasks.remove(task.id)
                inserted_count += 1

        builder.rebuild_all_schedules(insert_charging=True)
        return inserted_count


class RandomBestRepair(RepairOperator):
    """
    Random selection + Best quality
    Random task order, best position insertion
    """

    def __init__(self):
        super().__init__("RandomBestRepair")

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Random order, best position"""
        inserted_count = 0
        builder = ScheduleBuilder(solution)

        tasks_to_insert = [
            solution.task_dict[tid] for tid in removed_tasks
            if tid in solution.task_dict
        ]
        random.shuffle(tasks_to_insert)

        for task in tasks_to_insert:
            best_agv = None
            best_position = -1
            best_cost = float('inf')

            for agv in solution.agvs:
                for pos in range(len(agv.task_sequence) + 1):
                    agv.task_sequence.insert(pos, task.id)
                    is_feasible, _ = builder.build_complete_schedule(agv, insert_charging=True)

                    if is_feasible:
                        cost = agv.get_makespan() + agv.get_total_charging_time()
                        if cost < best_cost:
                            best_cost = cost
                            best_agv = agv
                            best_position = pos

                    agv.task_sequence.pop(pos)

            if best_agv is not None:
                best_agv.task_sequence.insert(best_position, task.id)
                solution.task_to_agv[task.id] = best_agv.id
                if task.id in solution.unassigned_tasks:
                    solution.unassigned_tasks.remove(task.id)
                inserted_count += 1

        builder.rebuild_all_schedules(insert_charging=True)
        return inserted_count


class RandomRandomRepair(RepairOperator):
    """
    Random selection + Random insertion
    Full randomization for maximum diversity
    """

    def __init__(self):
        super().__init__("RandomRandomRepair")

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Random order, random position"""
        inserted_count = 0
        builder = ScheduleBuilder(solution)

        tasks_to_insert = removed_tasks.copy()
        random.shuffle(tasks_to_insert)

        for task_id in tasks_to_insert:
            if task_id not in solution.task_dict:
                continue

            # Randomly select AGV
            agv = random.choice(solution.agvs)

            # Randomly select position
            pos = random.randint(0, len(agv.task_sequence))

            # Insert
            agv.task_sequence.insert(pos, task_id)
            solution.task_to_agv[task_id] = agv.id
            if task_id in solution.unassigned_tasks:
                solution.unassigned_tasks.remove(task_id)
            inserted_count += 1

        builder.rebuild_all_schedules(insert_charging=True)
        return inserted_count


class RandomTimeRepair(RepairOperator):
    """
    Random selection + Time optimization
    Random order, minimize makespan
    """

    def __init__(self):
        super().__init__("RandomTimeRepair")

    def repair(self, solution: AGVSolution, removed_tasks: List[int]) -> int:
        """Random order, minimize makespan"""
        inserted_count = 0
        builder = ScheduleBuilder(solution)

        tasks_to_insert = [
            solution.task_dict[tid] for tid in removed_tasks
            if tid in solution.task_dict
        ]
        random.shuffle(tasks_to_insert)

        for task in tasks_to_insert:
            best_agv = None
            best_position = -1
            best_makespan = float('inf')

            for agv in solution.agvs:
                for pos in range(len(agv.task_sequence) + 1):
                    agv.task_sequence.insert(pos, task.id)
                    is_feasible, _ = builder.build_complete_schedule(agv, insert_charging=True)

                    if is_feasible:
                        makespan = agv.get_makespan()
                        if makespan < best_makespan:
                            best_makespan = makespan
                            best_agv = agv
                            best_position = pos

                    agv.task_sequence.pop(pos)

            if best_agv is not None:
                best_agv.task_sequence.insert(best_position, task.id)
                solution.task_to_agv[task.id] = best_agv.id
                if task.id in solution.unassigned_tasks:
                    solution.unassigned_tasks.remove(task.id)
                inserted_count += 1

        builder.rebuild_all_schedules(insert_charging=True)
        return inserted_count


# Registry of all repair operators
REPAIR_OPERATORS = {
    'greedy_best': GreedyBestRepair,
    'greedy_random': GreedyRandomRepair,
    'greedy_time': GreedyTimeRepair,
    'adaptive_best': AdaptiveBestRepair,
    'adaptive_random': AdaptiveRandomRepair,
    'adaptive_time': AdaptiveTimeRepair,
    'random_best': RandomBestRepair,
    'random_random': RandomRandomRepair,
    'random_time': RandomTimeRepair
}


def get_repair_operator(name: str) -> RepairOperator:
    """Get repair operator by name"""
    if name not in REPAIR_OPERATORS:
        raise ValueError(f"Unknown repair operator: {name}")
    return REPAIR_OPERATORS[name]()
