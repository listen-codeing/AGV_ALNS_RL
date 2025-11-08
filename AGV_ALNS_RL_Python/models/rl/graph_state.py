"""
Graph representation for AGV solution state

Converts AGV solution into graph structure for GNN processing
"""
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from ..problem import AGVSolution, EventType


class GraphState:
    """
    Represents AGV solution as a graph for RL

    Nodes: Tasks, charging stations, depot
    Edges: Task sequences, charging relationships
    Features: Location, time, energy, route info
    """

    def __init__(self, solution: AGVSolution):
        """
        Initialize graph state from solution

        Args:
            solution: AGV solution to convert to graph
        """
        self.solution = solution
        self.num_tasks = len(solution.tasks)
        self.num_stations = len(solution.charging_stations)
        self.num_agvs = len(solution.agvs)

        # Build graph components
        self.node_features = None
        self.edge_index = None
        self.edge_features = None
        self.global_features = None

        self._build_graph()

    def _build_graph(self):
        """Build graph representation from solution"""
        # Node features: [task_id, agv_id, position_in_route, time_window, energy, location_x, location_y]
        node_features = []

        # Track node indices
        self.task_node_map = {}  # task_id -> node_index
        node_idx = 0

        # Add task nodes
        for task in self.solution.tasks:
            agv_id = self.solution.task_to_agv.get(task.id, -1)  # -1 if unassigned

            # Get position in route
            position = -1
            if agv_id >= 0:
                agv = self.solution.agv_dict[agv_id]
                if task.id in agv.task_sequence:
                    position = agv.task_sequence.index(task.id)

            # Normalize features
            time_window_width = task.latest_start - task.earliest_start
            time_window_center = (task.latest_start + task.earliest_start) / 2.0

            features = [
                float(task.id) / max(1, self.num_tasks),  # Normalized task ID
                float(agv_id) / max(1, self.num_agvs),    # Normalized AGV ID
                float(position) / max(1, 20),              # Normalized position
                time_window_center / 1000.0,               # Normalized time
                time_window_width / 1000.0,                # Normalized time window
                task.energy_consumption / 100.0,           # Normalized energy
                task.location[0] / 100.0,                  # Normalized x
                task.location[1] / 100.0                   # Normalized y
            ]

            node_features.append(features)
            self.task_node_map[task.id] = node_idx
            node_idx += 1

        self.node_features = torch.tensor(node_features, dtype=torch.float32)

        # Build edges based on task sequences and proximity
        edge_list = []
        edge_features_list = []

        # Add sequence edges (task -> next task in same route)
        for agv in self.solution.agvs:
            for i in range(len(agv.task_sequence) - 1):
                current_task_id = agv.task_sequence[i]
                next_task_id = agv.task_sequence[i + 1]

                current_node = self.task_node_map[current_task_id]
                next_node = self.task_node_map[next_task_id]

                # Bidirectional edges
                edge_list.append([current_node, next_node])
                edge_list.append([next_node, current_node])

                # Edge features: [edge_type, distance, time_diff]
                current_task = self.solution.task_dict[current_task_id]
                next_task = self.solution.task_dict[next_task_id]

                distance = np.sqrt(
                    (next_task.location[0] - current_task.location[0])**2 +
                    (next_task.location[1] - current_task.location[1])**2
                )

                time_diff = next_task.earliest_start - current_task.earliest_start

                edge_feat = [
                    1.0,  # Sequence edge
                    distance / 100.0,
                    time_diff / 1000.0
                ]

                edge_features_list.append(edge_feat)
                edge_features_list.append(edge_feat)

        # Add proximity edges (spatial relationships)
        # Connect tasks that are geographically close
        for i, task1 in enumerate(self.solution.tasks):
            for j, task2 in enumerate(self.solution.tasks):
                if i >= j:
                    continue

                distance = np.sqrt(
                    (task2.location[0] - task1.location[0])**2 +
                    (task2.location[1] - task1.location[1])**2
                )

                # Only connect nearby tasks
                if distance < 20.0:  # Threshold
                    node1 = self.task_node_map[task1.id]
                    node2 = self.task_node_map[task2.id]

                    edge_list.append([node1, node2])
                    edge_list.append([node2, node1])

                    edge_feat = [
                        0.0,  # Proximity edge
                        distance / 100.0,
                        0.0
                    ]

                    edge_features_list.append(edge_feat)
                    edge_features_list.append(edge_feat)

        # Convert to tensors
        if edge_list:
            self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            self.edge_features = torch.tensor(edge_features_list, dtype=torch.float32)
        else:
            # Empty graph case
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_features = torch.zeros((0, 3), dtype=torch.float32)

        # Global features: [makespan, total_charging, utilization, num_assigned, feasibility]
        makespan_norm = self.solution.makespan / 1000.0
        charging_norm = self.solution.total_charging_time / 1000.0
        utilization = len(self.solution.task_to_agv) / max(1, self.num_tasks)
        feasibility = 1.0 if self.solution.is_feasible else 0.0

        self.global_features = torch.tensor([
            makespan_norm,
            charging_norm,
            utilization,
            feasibility
        ], dtype=torch.float32)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get graph data for batch processing

        Returns:
            (node_features, edge_index, edge_features, global_features)
        """
        return self.node_features, self.edge_index, self.edge_features, self.global_features

    def to(self, device: torch.device):
        """Move tensors to device"""
        self.node_features = self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_features = self.edge_features.to(device)
        self.global_features = self.global_features.to(device)
        return self


def create_batch_from_solutions(solutions: List[AGVSolution]) -> Tuple:
    """
    Create batched graph data from multiple solutions

    Args:
        solutions: List of AGV solutions

    Returns:
        Batched graph tensors
    """
    graph_states = [GraphState(sol) for sol in solutions]

    # Concatenate node features
    all_node_features = []
    all_edge_indices = []
    all_edge_features = []
    all_global_features = []

    node_offset = 0

    for graph_state in graph_states:
        all_node_features.append(graph_state.node_features)

        # Offset edge indices for batching
        edge_index_offset = graph_state.edge_index + node_offset
        all_edge_indices.append(edge_index_offset)

        all_edge_features.append(graph_state.edge_features)
        all_global_features.append(graph_state.global_features.unsqueeze(0))

        node_offset += graph_state.node_features.shape[0]

    # Concatenate all
    batched_node_features = torch.cat(all_node_features, dim=0)
    batched_edge_index = torch.cat(all_edge_indices, dim=1)
    batched_edge_features = torch.cat(all_edge_features, dim=0)
    batched_global_features = torch.cat(all_global_features, dim=0)

    # Create batch assignment
    batch = []
    for i, graph_state in enumerate(graph_states):
        batch.extend([i] * graph_state.node_features.shape[0])
    batch = torch.tensor(batch, dtype=torch.long)

    return (
        batched_node_features,
        batched_edge_index,
        batched_edge_features,
        batched_global_features,
        batch
    )
