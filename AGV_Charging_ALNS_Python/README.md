# AGV Charging ALNS - Python Conversion

## Overview

This directory contains the **step-by-step conversion** of the AGV charging scheduling problem from C++ to Python.

**Original C++ code**: `Sloving AGV charging problem by ALNS (C++)/`

## Conversion Progress

### ✅ Phase 1: Basic Data Classes (COMPLETED)

#### 1. Schedule Event (`core/schedule_event.py`)

**Converted from**: `AGV.h` lines 14-44

**C++ Original**:
```cpp
enum class ScheduleEventType {
    TASK,           // 作业事件
    CHARGING        // 充电事件
};

struct ScheduleEvent {
    ScheduleEventType type;
    int task_id;
    double start_time;
    double end_time;
    double arrival_time;
    double waiting_time;
    ...
};
```

**Python Equivalent**:
```python
class ScheduleEventType(Enum):
    TASK = "TASK"
    CHARGING = "CHARGING"

@dataclass
class ScheduleEvent:
    type: ScheduleEventType
    task_id: int
    start_time: float
    end_time: float
    arrival_time: float
    waiting_time: float
```

**Key Features**:
- Enum for event types
- Dataclass for automatic `__init__`, `__repr__`, etc.
- `update_waiting_time()` method preserved
- Properties for computed values (e.g., `duration`)

---

#### 2. AGV Class (`core/agv.py`)

**Converted from**: `AGV.h` lines 46-116 and `AGV.cpp`

**C++ Original**:
```cpp
class AGV {
public:
    Parameter* pm;
    int truck_id;
    double initial_soc;
    int container_num;

    std::vector<Task*> task_sequence;
    std::vector<double> task_start_times;
    std::vector<double> task_end_times;

    std::vector<bool> isCharge;
    std::vector<std::pair<int, int>> charging_sessions;

    std::vector<double> soc_after_task;
    std::vector<double> soc_before_task;

    std::vector<ScheduleEvent> schedule_events;

    void update_schedule_from_event(int event_index, double new_start_time, double xi_uncertainty);
    double get_makespan() const;
    ...
};
```

**Python Equivalent**:
```python
class AGV:
    def __init__(self, truck_id: int = 0, initial_soc: float = 1.0, parameter=None):
        self.truck_id: int = truck_id
        self.initial_soc: float = initial_soc

        self.task_sequence: List = []
        self.task_start_times: List[float] = []

        self.isCharge: List[bool] = []
        self.charging_sessions: List[Tuple[int, int]] = []

        self.soc_after_task: List[float] = []
        self.soc_before_task: List[float] = []

        self.schedule_events: List[ScheduleEvent] = []

    def update_schedule_from_event(self, event_index: int,
                                   new_start_time: float,
                                   xi_uncertainty: float = 0.0):
        ...

    def get_makespan(self) -> float:
        ...
```

**Key Features**:
- All C++ member variables converted to Python attributes
- Type hints for clarity
- All methods preserved:
  - `generate_task_sequence()`
  - `reset()`
  - `validate_schedule()`
  - `get_travel_time()`
  - `get_charging_time()`
  - `update_schedule_from_event()` - full implementation
  - `calculate_remaining_time_from_event()`
  - `get_makespan()`
- Placeholder comments where Task/Parameter classes will be integrated

---

## Design Decisions

### 1. **Type Hints**
All variables and function signatures include type hints for better IDE support and code clarity.

### 2. **Dataclasses**
Used `@dataclass` for simple data structures like `ScheduleEvent` to reduce boilerplate.

### 3. **Enums**
Python's `Enum` class used instead of C++ `enum class`.

### 4. **Lists instead of std::vector**
Python lists are used with type hints: `List[float]`, `List[bool]`, etc.

### 5. **Tuples instead of std::pair**
`charging_sessions: List[Tuple[int, int]]` instead of `std::vector<std::pair<int, int>>`

### 6. **Forward Compatibility**
Comments and placeholders added where future classes (Task, Parameter) will be integrated.

---

## File Structure

```
AGV_Charging_ALNS_Python/
├── core/
│   ├── __init__.py           # Package initialization
│   ├── schedule_event.py     # ScheduleEvent and ScheduleEventType
│   └── agv.py                # AGV class
├── README.md                  # This file
└── ...                        # Future additions
```

---

## Next Steps

### Phase 2: Problem Definition Classes
- [ ] Find and convert `Task` class
- [ ] Find and convert `Parameter` class
- [ ] Convert any other base data structures

### Phase 3: Solution Class
- [ ] Convert `AGV_solution` class from `Heuristic/AGV_solution.h`

### Phase 4: ALNS Framework
- [ ] Convert operator classes
- [ ] Convert ALNS main algorithm

---

## Code Quality

✅ **Direct C++ to Python mapping**: Each method corresponds to its C++ counterpart
✅ **Comments in Chinese preserved**: Original Chinese comments kept for clarity
✅ **Full documentation**: Docstrings explain correspondence to C++ code
✅ **Type safety**: Type hints for all variables and methods

---

## Usage Example (Coming Soon)

```python
from core.agv import AGV
from core.schedule_event import ScheduleEvent, ScheduleEventType

# Create an AGV
agv = AGV(truck_id=0, initial_soc=1.0)

# Add a task event
event = ScheduleEvent(
    type=ScheduleEventType.TASK,
    task_id=0,
    start_time=10.0,
    end_time=15.0
)
agv.schedule_events.append(event)

# Get makespan
print(f"Makespan: {agv.get_makespan()}")
```

---

## Questions or Issues?

If you notice any discrepancies with the C++ code or have suggestions for improvements, please let me know!

---

**Conversion Date**: 2025-11-08
**Converted By**: Claude (AI Assistant)
**Original C++ Author**: limin
