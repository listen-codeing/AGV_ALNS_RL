"""
Visualization tools for AGV charging scheduling solutions
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
from typing import List, Optional, Dict
import seaborn as sns

from ..models.problem import AGVSolution, EventType


def plot_solution_gantt(solution: AGVSolution, save_path: Optional[str] = None):
    """
    Create a Gantt chart visualization of the AGV schedules

    Args:
        solution: AGV solution to visualize
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    event_colors = {
        EventType.TASK: colors[0],
        EventType.CHARGING: colors[1],
        EventType.TRAVEL: colors[2],
        EventType.IDLE: colors[3]
    }

    # Plot each AGV's schedule
    for i, agv in enumerate(solution.agvs):
        if not agv.schedule_events:
            continue

        for event in agv.schedule_events:
            start = event.start_time
            duration = event.end_time - event.start_time
            color = event_colors[event.event_type]

            # Draw bar
            ax.barh(i, duration, left=start, height=0.8, color=color, alpha=0.7, edgecolor='black')

            # Add label
            if event.event_type == EventType.TASK:
                label = f"T{event.task_id}"
                ax.text(start + duration/2, i, label, ha='center', va='center', fontsize=8)
            elif event.event_type == EventType.CHARGING:
                label = f"CS{event.station_id}"
                ax.text(start + duration/2, i, label, ha='center', va='center', fontsize=8)

    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('AGV', fontsize=12)
    ax.set_yticks(range(len(solution.agvs)))
    ax.set_yticklabels([f'AGV {agv.id}' for agv in solution.agvs])
    ax.set_title(f'AGV Schedule (Makespan: {solution.makespan:.2f})', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=event_colors[EventType.TASK], label='Task', alpha=0.7),
        mpatches.Patch(color=event_colors[EventType.CHARGING], label='Charging', alpha=0.7),
        mpatches.Patch(color=event_colors[EventType.TRAVEL], label='Travel', alpha=0.7),
        mpatches.Patch(color=event_colors[EventType.IDLE], label='Idle', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Gantt chart to {save_path}")

    plt.show()


def plot_solution_map(solution: AGVSolution, save_path: Optional[str] = None):
    """
    Create a 2D map visualization showing task locations and AGV routes

    Args:
        solution: AGV solution to visualize
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(solution.agvs)))

    # Plot charging stations
    for station in solution.charging_stations:
        ax.scatter(station.location[0], station.location[1],
                  s=500, c='red', marker='s', alpha=0.5, edgecolors='black', linewidth=2)
        ax.text(station.location[0], station.location[1],
               f'CS{station.id}', ha='center', va='center', fontsize=10, fontweight='bold')

    # Plot tasks
    for task in solution.tasks:
        ax.scatter(task.location[0], task.location[1],
                  s=200, c='lightblue', marker='o', edgecolors='black', linewidth=1)
        ax.text(task.location[0] + 2, task.location[1] + 2,
               f'T{task.id}', fontsize=8)

    # Plot AGV routes
    for i, agv in enumerate(solution.agvs):
        if not agv.task_sequence:
            continue

        route_x = [agv.initial_location[0]]
        route_y = [agv.initial_location[1]]

        for task_id in agv.task_sequence:
            task = solution.task_dict[task_id]
            route_x.append(task.location[0])
            route_y.append(task.location[1])

        ax.plot(route_x, route_y, c=colors[i], linewidth=2, alpha=0.6,
               label=f'AGV {agv.id}', marker='>', markersize=5)

    # Formatting
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('AGV Routes and Task Locations', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved route map to {save_path}")

    plt.show()


def plot_convergence(cost_history: List[float], best_cost_history: List[float],
                    save_path: Optional[str] = None):
    """
    Plot ALNS convergence history

    Args:
        cost_history: Current solution cost over iterations
        best_cost_history: Best solution cost over iterations
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = range(len(cost_history))

    ax.plot(iterations, cost_history, alpha=0.5, label='Current Solution', linewidth=1)
    ax.plot(iterations, best_cost_history, label='Best Solution', linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('ALNS Convergence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")

    plt.show()


def plot_operator_usage(operator_history: List[tuple], operator_names: Dict[str, List[str]],
                       save_path: Optional[str] = None):
    """
    Plot operator usage statistics

    Args:
        operator_history: List of (destroy_op, repair_op) tuples
        operator_names: Dict with 'destroy' and 'repair' operator names
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Count operator usage
    destroy_counts = {}
    repair_counts = {}

    for destroy_op, repair_op in operator_history:
        destroy_counts[destroy_op] = destroy_counts.get(destroy_op, 0) + 1
        repair_counts[repair_op] = repair_counts.get(repair_op, 0) + 1

    # Plot destroy operators
    destroy_ops = list(destroy_counts.keys())
    destroy_vals = list(destroy_counts.values())
    ax1.bar(range(len(destroy_ops)), destroy_vals, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(destroy_ops)))
    ax1.set_xticklabels(destroy_ops, rotation=45, ha='right')
    ax1.set_ylabel('Usage Count', fontsize=12)
    ax1.set_title('Destroy Operator Usage', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Plot repair operators
    repair_ops = list(repair_counts.keys())
    repair_vals = list(repair_counts.values())
    ax2.bar(range(len(repair_ops)), repair_vals, color='coral', alpha=0.7)
    ax2.set_xticks(range(len(repair_ops)))
    ax2.set_xticklabels(repair_ops, rotation=45, ha='right')
    ax2.set_ylabel('Usage Count', fontsize=12)
    ax2.set_title('Repair Operator Usage', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved operator usage plot to {save_path}")

    plt.show()


def plot_charging_station_utilization(solution: AGVSolution, save_path: Optional[str] = None):
    """
    Plot charging station utilization over time

    Args:
        solution: AGV solution
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(solution.charging_stations)))

    for i, station in enumerate(solution.charging_stations):
        for event in station.events:
            ax.barh(i, event.end_time - event.start_time,
                   left=event.start_time, height=0.8,
                   color=colors[i], alpha=0.6, edgecolor='black')
            ax.text(event.start_time + (event.end_time - event.start_time)/2, i,
                   f'AGV{event.agv_id}', ha='center', va='center', fontsize=8)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Charging Station', fontsize=12)
    ax.set_yticks(range(len(solution.charging_stations)))
    ax.set_yticklabels([f'Station {s.id}' for s in solution.charging_stations])
    ax.set_title('Charging Station Utilization', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add capacity lines
    for i, station in enumerate(solution.charging_stations):
        ax.axhline(y=i + 0.5, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved station utilization plot to {save_path}")

    plt.show()


def plot_battery_soc(solution: AGVSolution, save_path: Optional[str] = None):
    """
    Plot battery State of Charge (SOC) over time for each AGV

    Args:
        solution: AGV solution
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(len(solution.agvs), 1,
                            figsize=(14, 4*len(solution.agvs)),
                            sharex=True)

    if len(solution.agvs) == 1:
        axes = [axes]

    for i, agv in enumerate(solution.agvs):
        if not agv.schedule_events:
            continue

        times = []
        socs = []

        for event in agv.schedule_events:
            times.append(event.start_time)
            socs.append(event.soc_before)
            times.append(event.end_time)
            socs.append(event.soc_after)

        axes[i].plot(times, socs, marker='o', linewidth=2, markersize=4)
        axes[i].axhline(y=agv.min_soc, color='red', linestyle='--',
                       label=f'Min SOC ({agv.min_soc})')
        axes[i].fill_between(times, 0, agv.min_soc, alpha=0.2, color='red')

        axes[i].set_ylabel('SOC', fontsize=11)
        axes[i].set_title(f'AGV {agv.id} Battery Level', fontsize=12, fontweight='bold')
        axes[i].grid(alpha=0.3)
        axes[i].legend(loc='lower left')
        axes[i].set_ylim([0, 1.05])

    axes[-1].set_xlabel('Time', fontsize=12)
    fig.suptitle('Battery State of Charge Over Time', fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SOC plot to {save_path}")

    plt.show()


def create_summary_report(solution: AGVSolution, output_dir: str = 'results'):
    """
    Create a comprehensive visualization report

    Args:
        solution: AGV solution to visualize
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Generating visualization report...")

    # Gantt chart
    plot_solution_gantt(solution, save_path=f'{output_dir}/gantt_chart.png')

    # Route map
    plot_solution_map(solution, save_path=f'{output_dir}/route_map.png')

    # Station utilization
    plot_charging_station_utilization(solution, save_path=f'{output_dir}/station_utilization.png')

    # Battery SOC
    plot_battery_soc(solution, save_path=f'{output_dir}/battery_soc.png')

    # Save solution summary
    summary = solution.get_solution_summary()
    with open(f'{output_dir}/solution_summary.txt', 'w') as f:
        f.write("AGV CHARGING SCHEDULING SOLUTION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Makespan:                 {summary['makespan']:.2f}\n")
        f.write(f"Total Charging Time:      {summary['total_charging_time']:.2f}\n")
        f.write(f"Total Idle Time:          {summary['total_idle_time']:.2f}\n")
        f.write(f"Total Energy Cost:        {summary['total_energy_cost']:.2f}\n")
        f.write(f"Number of Tasks:          {summary['num_tasks']}\n")
        f.write(f"Assigned Tasks:           {summary['num_assigned_tasks']}\n")
        f.write(f"Unassigned Tasks:         {summary['num_unassigned_tasks']}\n")
        f.write(f"Feasible:                 {summary['is_feasible']}\n")
        f.write(f"AGVs Used:                {summary['num_agvs_used']}\n\n")

        f.write("AGV Details:\n")
        f.write("-" * 60 + "\n")
        for agv_summary in summary['agv_summaries']:
            f.write(f"  AGV {agv_summary['id']}:\n")
            f.write(f"    Tasks:          {agv_summary['num_tasks']}\n")
            f.write(f"    Makespan:       {agv_summary['makespan']:.2f}\n")
            f.write(f"    Charging Time:  {agv_summary['charging_time']:.2f}\n")
            f.write(f"    Idle Time:      {agv_summary['idle_time']:.2f}\n")
            f.write(f"    Num Charges:    {agv_summary['num_charges']}\n")
            f.write(f"    Final SOC:      {agv_summary['final_soc']:.2f}\n\n")

        f.write("Charging Station Utilization:\n")
        f.write("-" * 60 + "\n")
        for station_info in summary['station_utilizations']:
            f.write(f"  Station {station_info['id']}:\n")
            f.write(f"    Utilization:    {station_info['utilization']*100:.1f}%\n")
            f.write(f"    Peak Usage:     {station_info['peak_usage']}\n")
            f.write(f"    Events:         {station_info['num_events']}\n\n")

    print(f"Report saved to {output_dir}/")
