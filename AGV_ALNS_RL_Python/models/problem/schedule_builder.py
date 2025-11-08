"""
Schedule builder for AGV charging scheduling problem

Converts task sequences and charging decisions into complete feasible schedules
"""
from typing import List, Tuple, Optional, Dict
import numpy as np
from .agv import AGV, EventType, ScheduleEvent
from .task import Task
from .charging_station import ChargingStation
from .solution import AGVSolution


class ScheduleBuilder:
    """
    Builds complete schedules for AGVs including:
    - Task execution
    - Travel between locations
    - Charging sessions
    - Idle/waiting periods
    """

    def __init__(self, solution: AGVSolution):
        """
        Initialize schedule builder

        Args:
            solution: The AGV solution to build schedules for
        """
        self.solution = solution

    def build_complete_schedule(
        self,
        agv: AGV,
        insert_charging: bool = True,
        charging_threshold: float = 0.3
    ) -> Tuple[bool, List[str]]:
        """
        Build complete schedule for a single AGV

        Args:
            agv: The AGV to build schedule for
            insert_charging: Whether to automatically insert charging sessions
            charging_threshold: SOC threshold below which charging is triggered

        Returns:
            (is_feasible, list_of_issues)
        """
        issues = []

        # Reset AGV schedule
        agv.schedule_events = []
        current_time = 0.0
        current_location = agv.initial_location
        current_soc = agv.initial_soc

        # Process each task in sequence
        for task_id in agv.task_sequence:
            task = self.solution.task_dict.get(task_id)
            if task is None:
                issues.append(f"Task {task_id} not found")
                continue

            # Check if charging is needed before this task
            if insert_charging:
                # Estimate energy needed for travel + task
                travel_energy = agv.get_travel_energy(current_location, task.location)
                task_energy = task.energy_consumption
                total_energy_needed = travel_energy + task_energy

                # Check if we need to charge
                if not agv.is_energy_feasible(total_energy_needed, current_soc) or \
                   current_soc < charging_threshold:
                    # Find best charging station and insert charging
                    charging_result = self._insert_optimal_charging(
                        agv, current_location, current_time, current_soc,
                        target_soc=max(0.8, current_soc + 0.3)  # Charge to reasonable level
                    )

                    if charging_result is not None:
                        charging_event, new_time, new_soc = charging_result
                        agv.schedule_events.append(charging_event)
                        current_time = new_time
                        current_soc = new_soc
                        current_location = charging_event.location
                    else:
                        issues.append(f"Could not find charging opportunity before task {task_id}")

            # Add travel to task location
            if current_location != task.location:
                travel_event, new_time, new_soc = self._create_travel_event(
                    agv, current_location, task.location, current_time, current_soc
                )
                agv.schedule_events.append(travel_event)
                current_time = new_time
                current_soc = new_soc
                current_location = task.location

            # Add idle/waiting time if necessary (task time window)
            if current_time < task.earliest_start:
                idle_event = ScheduleEvent(
                    event_type=EventType.IDLE,
                    start_time=current_time,
                    end_time=task.earliest_start,
                    location=current_location,
                    soc_before=current_soc,
                    soc_after=current_soc,  # Assuming minimal consumption during idle
                    energy_change=-agv.idle_consumption * (task.earliest_start - current_time)
                )
                agv.schedule_events.append(idle_event)
                current_time = task.earliest_start
                current_soc = agv.get_soc_after_consumption(
                    agv.idle_consumption * idle_event.duration(),
                    current_soc
                )

            # Check time window feasibility
            if current_time > task.latest_start:
                issues.append(
                    f"Task {task_id}: Arrival time {current_time:.2f} > "
                    f"latest start {task.latest_start:.2f}"
                )

            # Add task execution event
            task_start = max(current_time, task.earliest_start)
            task_end = task_start + task.duration

            # Check energy feasibility
            if not agv.is_energy_feasible(task.energy_consumption, current_soc):
                issues.append(
                    f"Task {task_id}: Insufficient energy "
                    f"(need {task.energy_consumption:.2f}, SOC={current_soc:.2f})"
                )

            task_event = ScheduleEvent(
                event_type=EventType.TASK,
                start_time=task_start,
                end_time=task_end,
                location=task.location,
                task_id=task_id,
                energy_change=-task.energy_consumption,
                soc_before=current_soc,
                soc_after=agv.get_soc_after_consumption(task.energy_consumption, current_soc)
            )
            agv.schedule_events.append(task_event)
            current_time = task_end
            current_soc = task_event.soc_after
            current_location = task.location

        # Update AGV state
        agv.current_time = current_time
        agv.current_location = current_location
        agv.current_soc = current_soc

        is_feasible = len(issues) == 0
        return is_feasible, issues

    def _create_travel_event(
        self,
        agv: AGV,
        from_loc: Tuple[float, float],
        to_loc: Tuple[float, float],
        start_time: float,
        current_soc: float
    ) -> Tuple[ScheduleEvent, float, float]:
        """Create a travel event between two locations"""
        travel_time = agv.get_travel_time(from_loc, to_loc)
        travel_energy = agv.get_travel_energy(from_loc, to_loc)
        end_time = start_time + travel_time
        new_soc = agv.get_soc_after_consumption(travel_energy, current_soc)

        event = ScheduleEvent(
            event_type=EventType.TRAVEL,
            start_time=start_time,
            end_time=end_time,
            location=to_loc,  # Destination location
            energy_change=-travel_energy,
            soc_before=current_soc,
            soc_after=new_soc
        )

        return event, end_time, new_soc

    def _insert_optimal_charging(
        self,
        agv: AGV,
        current_location: Tuple[float, float],
        current_time: float,
        current_soc: float,
        target_soc: float = 0.8
    ) -> Optional[Tuple[ScheduleEvent, float, float]]:
        """
        Find and insert an optimal charging session

        Args:
            agv: The AGV that needs charging
            current_location: Current location of the AGV
            current_time: Current time
            current_soc: Current state of charge
            target_soc: Target SOC after charging

        Returns:
            (charging_event, new_time, new_soc) or None if no feasible charging found
        """
        best_station = None
        best_cost = float('inf')
        best_arrival_time = 0.0

        # Evaluate each charging station
        for station in self.solution.charging_stations:
            # Calculate travel to station
            travel_time = agv.get_travel_time(current_location, station.location)
            travel_energy = agv.get_travel_energy(current_location, station.location)
            arrival_time = current_time + travel_time
            arrival_soc = agv.get_soc_after_consumption(travel_energy, current_soc)

            # Check if we can reach the station
            if arrival_soc < agv.min_soc:
                continue

            # Calculate charging amount and time
            energy_to_charge = agv.calculate_energy_to_soc(target_soc, arrival_soc)
            charging_time = agv.calculate_charging_time(energy_to_charge, station.charging_rate)

            # Check station availability
            if not station.is_available(arrival_time, arrival_time + charging_time):
                continue

            # Calculate cost (prefer closer stations, less waiting)
            cost = travel_time + charging_time

            if cost < best_cost:
                best_cost = cost
                best_station = station
                best_arrival_time = arrival_time

        if best_station is None:
            return None

        # Create charging event
        travel_time = agv.get_travel_time(current_location, best_station.location)
        travel_energy = agv.get_travel_energy(current_location, best_station.location)
        arrival_soc = agv.get_soc_after_consumption(travel_energy, current_soc)

        energy_to_charge = agv.calculate_energy_to_soc(target_soc, arrival_soc)
        charging_time = agv.calculate_charging_time(energy_to_charge, best_station.charging_rate)

        charging_start = current_time + travel_time
        charging_end = charging_start + charging_time
        final_soc = agv.get_soc_after_charging(energy_to_charge, arrival_soc)

        # Add charging event to station
        best_station.add_charging_event(agv.id, charging_start, charging_end, energy_to_charge)

        # Create composite event (travel + charging)
        # For simplicity, we create just the charging event
        # (travel is handled separately in the main loop)
        charging_event = ScheduleEvent(
            event_type=EventType.CHARGING,
            start_time=charging_start,
            end_time=charging_end,
            location=best_station.location,
            station_id=best_station.id,
            energy_change=energy_to_charge,
            soc_before=arrival_soc,
            soc_after=final_soc
        )

        return charging_event, charging_end, final_soc

    def rebuild_all_schedules(self, insert_charging: bool = True) -> bool:
        """
        Rebuild schedules for all AGVs in the solution

        Args:
            insert_charging: Whether to automatically insert charging sessions

        Returns:
            True if all schedules are feasible, False otherwise
        """
        all_feasible = True

        # Clear all charging events from stations
        for station in self.solution.charging_stations:
            station.events = []

        # Build schedule for each AGV
        for agv in self.solution.agvs:
            if len(agv.task_sequence) == 0:
                continue

            is_feasible, issues = self.build_complete_schedule(agv, insert_charging)

            if not is_feasible:
                all_feasible = False
                self.solution.violations.extend(issues)

        # Update solution metrics
        self.solution.update_metrics()

        return all_feasible
