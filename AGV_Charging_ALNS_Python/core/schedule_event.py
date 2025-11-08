"""
Schedule Event data structures
Converted from C++ ScheduleEvent in AGV.h
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ScheduleEventType(Enum):
    """
    Type of schedule event

    Corresponds to C++ enum:
    enum class ScheduleEventType {
        TASK,      // 作业事件
        CHARGING   // 充电事件
    };
    """
    TASK = "TASK"           # 作业事件
    CHARGING = "CHARGING"    # 充电事件


@dataclass
class ScheduleEvent:
    """
    Represents a single event in AGV's schedule

    Corresponds to C++ struct ScheduleEvent in AGV.h

    Attributes:
        type: Event type (TASK or CHARGING)
        task_id: ID of the task (for both TASK and CHARGING events)
        start_time: When the event starts
        end_time: When the event ends
        arrival_time: When AGV arrives (equals start_time for tasks,
                      may be earlier for charging)
        waiting_time: Time spent waiting before event starts (for charging)
    """
    type: ScheduleEventType
    task_id: int
    start_time: float
    end_time: float
    arrival_time: float = None  # Will be set in __post_init__
    waiting_time: float = 0.0

    def __post_init__(self):
        """
        Initialize arrival_time if not provided

        Corresponds to C++ constructors:
        - ScheduleEvent(type, id, start, end): arrival_time = start
        - ScheduleEvent(type, id, arrival, start, end): arrival_time provided
        """
        if self.arrival_time is None:
            self.arrival_time = self.start_time

        # Update waiting time
        self.update_waiting_time()

    def update_waiting_time(self):
        """
        Update waiting time (for charging events)

        Corresponds to C++ method:
        void update_waiting_time() {
            if (type == ScheduleEventType::CHARGING) {
                waiting_time = start_time - arrival_time;
            }
        }
        """
        if self.type == ScheduleEventType.CHARGING:
            self.waiting_time = self.start_time - self.arrival_time

    @property
    def duration(self) -> float:
        """Get duration of this event"""
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        type_str = "TASK" if self.type == ScheduleEventType.TASK else "CHARGE"
        return (f"ScheduleEvent({type_str}, task_id={self.task_id}, "
                f"time=[{self.start_time:.2f}, {self.end_time:.2f}])")
