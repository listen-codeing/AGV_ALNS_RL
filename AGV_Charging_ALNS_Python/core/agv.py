"""
AGV (Automated Guided Vehicle) class
Converted from C++ AGV class in AGV.h and AGV.cpp
"""
from typing import List, Tuple, Optional
from .schedule_event import ScheduleEvent, ScheduleEventType


class AGV:
    """
    Represents an Automated Guided Vehicle

    Corresponds to C++ class AGV in AGV.h

    Attributes:
        truck_id: AGV编号 (AGV ID)
        initial_soc: 初始电量百分比 (initial state of charge, 0.0-1.0)
        container_num: 箱子数量 (number of containers/tasks)

        # Task sequence
        task_sequence: 任务序列 (sequence of Task pointers in C++)
        task_arrival_times: 每个任务的到达时间
        task_start_times: 每个任务的开始时间
        task_end_times: 每个任务的结束时间
        task_completion_times: 任务完成时间

        # Charging information
        isCharge: 每个任务后是否充电 (whether to charge after each task)
        charging_sessions: 充电会话 (station_id, session_id) pairs
        charging_start_times: 充电开始时间
        arrival_at_station: 到达充电站时间
        charging_end_times: 充电结束时间
        charging_durations: 充电持续时间

        # Battery state (SOC = State of Charge)
        soc_after_task: 每个任务后的电量百分比
        soc_before_task: 每个任务前的电量百分比
        soc_at_cs_arrival: 到达充电站时的电量
        soc_after_charging: 充电后的电量
        soc_charging_durations: 充了多少电

        # Statistics
        max_completion_time: 最大完成时间
        total_waiting_time: 总等待时间
        total_charging_time: 总充电时间
        total_travel_time: 总旅行时间

        # Gurobi original values (for warm start)
        original_Ts: 原始任务开始时间的解值
        original_Te: 原始任务结束时间的解值
        original_Se: 原始任务后 SOC 的解值
        original_Sr: 原始到充电站时 SOC 的解值
        original_Sf: 原始充电后 SOC 的解值

        # Unified schedule events
        schedule_events: 统一的调度事件序列
    """

    def __init__(self, truck_id: int = 0, initial_soc: float = 1.0, parameter=None):
        """
        Initialize AGV

        Corresponds to C++ constructor:
        AGV::AGV(Parameter &pm, int id)

        Args:
            truck_id: AGV ID
            initial_soc: Initial state of charge (0.0 to 1.0)
            parameter: Reference to Parameter object (optional for now)
        """
        # Store parameter reference (will be used later when Parameter class is defined)
        self.pm = parameter

        # Basic attributes
        self.truck_id: int = truck_id
        self.initial_soc: float = initial_soc
        self.container_num: int = 0

        # Task sequence - 任务序列
        # In C++: std::vector<Task*> task_sequence
        # In Python: we'll store task objects directly (to be defined later)
        self.task_sequence: List = []

        # Task timing - 任务时间信息
        self.task_arrival_times: List[float] = []
        self.task_start_times: List[float] = []
        self.task_end_times: List[float] = []
        self.task_completion_times: List[float] = []

        # Charging information - 充电信息
        self.isCharge: List[bool] = []
        self.charging_sessions: List[Tuple[int, int]] = []  # (station_id, session_id)
        self.charging_start_times: List[float] = []
        self.arrival_at_station: List[float] = []
        self.charging_end_times: List[float] = []
        self.charging_durations: List[float] = []

        # Battery state (SOC) - 电池状态
        self.soc_after_task: List[float] = []
        self.soc_before_task: List[float] = []
        self.soc_at_cs_arrival: List[float] = []
        self.soc_after_charging: List[float] = []
        self.soc_charging_durations: List[float] = []

        # Statistics - 统计信息
        self.max_completion_time: float = 0.0
        self.total_waiting_time: float = 0.0
        self.total_charging_time: float = 0.0
        self.total_travel_time: float = 0.0

        # Gurobi original values - 原始Gurobi求解值
        self.original_Ts: List[float] = []
        self.original_Te: List[float] = []
        self.original_Se: List[float] = []
        self.original_Sr: List[float] = []
        self.original_Sf: List[float] = []

        # Unified schedule events - 统一的调度事件
        self.schedule_events: List[ScheduleEvent] = []

    def generate_task_sequence(self):
        """
        Generate task sequence

        Corresponds to C++ method:
        void AGV::generate_task_sequence()

        Initializes timing arrays based on container_num
        """
        # Clear existing timing data
        self.task_start_times.clear()
        self.task_end_times.clear()

        # Resize arrays
        self.task_start_times = [0.0] * self.container_num
        self.task_end_times = [0.0] * self.container_num

    def reset(self):
        """
        Reset AGV to initial state

        Corresponds to C++ method:
        void AGV::reset()

        Clears all data structures
        """
        self.pm = None
        self.truck_id = 0
        self.initial_soc = 0.0
        self.container_num = 0

        # Clear task sequence
        self.task_sequence.clear()
        self.task_start_times.clear()
        self.task_end_times.clear()

        # Clear charging information
        self.isCharge.clear()
        self.charging_sessions.clear()
        self.charging_start_times.clear()

        # Clear SOC information
        self.soc_after_task.clear()
        self.soc_before_task.clear()
        self.soc_at_cs_arrival.clear()
        self.soc_after_charging.clear()

        # Reset statistics
        self.total_waiting_time = 0.0
        self.total_charging_time = 0.0
        self.total_travel_time = 0.0

        # Clear original values
        self.original_Ts.clear()
        self.original_Te.clear()
        self.original_Se.clear()
        self.original_Sr.clear()
        self.original_Sf.clear()

        # Clear schedule events
        self.schedule_events.clear()

    def validate_schedule(self) -> bool:
        """
        Validate whether the schedule satisfies all constraints

        Corresponds to C++ method:
        bool AGV::validate_schedule(const Parameter& params)

        Returns:
            True if schedule is valid, False otherwise
        """
        # TODO: Implement validation logic when Parameter class is available
        return True

    def get_travel_time(self) -> float:
        """
        Get total travel time

        Corresponds to C++ method:
        double AGV::get_travel_time()

        Returns:
            Total travel time
        """
        if self.task_end_times:
            self.total_travel_time = self.task_end_times[-1]
        return self.total_travel_time

    def get_charging_time(self) -> float:
        """
        Get total charging time

        Corresponds to C++ method:
        double AGV::get_charging_time()

        Returns:
            Total charging time
        """
        self.total_charging_time = 0.0
        # Sum up all charging durations
        for c_time in self.charging_durations:
            self.total_charging_time += c_time
        return self.total_charging_time

    def update_schedule_from_event(self, event_index: int, new_start_time: float,
                                   xi_uncertainty: float = 0.0):
        """
        Update schedule from a specific event

        Corresponds to C++ method:
        void AGV::update_schedule_from_event(int event_index, double new_start_time,
                                             double xi_uncertainty)

        Updates the specified event and propagates time changes to subsequent events

        Args:
            event_index: Index of the event to update
            new_start_time: New start time for the event
            xi_uncertainty: Time uncertainty parameter
        """
        if event_index >= len(self.schedule_events):
            return

        event = self.schedule_events[event_index]

        # Calculate time shift
        time_shift = new_start_time - event.start_time

        # If no time change, return early
        if abs(time_shift) < 1e-6:
            return

        # Update current event
        event.start_time = new_start_time

        if event.type == ScheduleEventType.CHARGING:
            # For charging event, update end time based on charging duration
            # In C++: event.end_time = new_start_time + pm->tu
            # TODO: Use actual charging duration when Parameter is available
            event.end_time = new_start_time + 1.0  # Placeholder

            # Update waiting time
            event.waiting_time = event.start_time - event.arrival_time

            # Update charging start times record
            for i, charge_time in enumerate(self.charging_start_times):
                if abs(charge_time - (new_start_time - time_shift)) < 1e-6:
                    self.charging_start_times[i] = new_start_time
                    break

        else:  # TASK event
            # For task event, update end time based on task duration
            # In C++: event.end_time = new_start_time + task_sequence[event.task_id]->tl + xi_uncertainty
            # TODO: Use actual task duration when Task class is available
            task_duration = 1.0  # Placeholder
            event.end_time = new_start_time + task_duration + xi_uncertainty

            # Update task timing records
            self.task_start_times[event.task_id] = event.start_time
            self.task_end_times[event.task_id] = event.end_time

        # Propagate time shift to subsequent events
        for i in range(event_index + 1, len(self.schedule_events)):
            next_event = self.schedule_events[i]

            # Shift all times
            next_event.start_time += time_shift
            next_event.end_time += time_shift

            if next_event.type == ScheduleEventType.CHARGING:
                next_event.arrival_time += time_shift
                # Waiting time remains unchanged (both start and arrival shift equally)

                # Update charging start times record
                for j, charge_time in enumerate(self.charging_start_times):
                    if abs(charge_time - (next_event.start_time - time_shift)) < 1e-6:
                        self.charging_start_times[j] = next_event.start_time
                        break

            else:  # TASK
                # Update task timing records
                self.task_start_times[next_event.task_id] = next_event.start_time
                self.task_end_times[next_event.task_id] = next_event.end_time

    def calculate_remaining_time_from_event(self, event_index: int) -> float:
        """
        Calculate remaining time from a specific event to the end

        Corresponds to C++ method:
        double AGV::calculate_remaining_time_from_event(int event_index) const

        Args:
            event_index: Index of the event

        Returns:
            Time from event start to schedule end
        """
        if event_index >= len(self.schedule_events):
            return 0.0

        start_time = self.schedule_events[event_index].start_time
        end_time = self.schedule_events[-1].end_time
        return end_time - start_time

    def get_makespan(self) -> float:
        """
        Get the makespan (total completion time)

        Corresponds to C++ method:
        double AGV::get_makespan() const

        Returns:
            End time of the last event, or 0 if no events
        """
        if not self.schedule_events:
            return 0.0
        return self.schedule_events[-1].end_time

    def __repr__(self) -> str:
        """String representation of AGV"""
        return (f"AGV(id={self.truck_id}, soc={self.initial_soc:.2f}, "
                f"tasks={len(self.task_sequence)}, events={len(self.schedule_events)})")
