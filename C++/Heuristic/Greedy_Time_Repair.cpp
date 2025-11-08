//
// Created by 张欢 on 2025/8/2.
//

#include "Greedy_Time_Repair.h"
#include "AGV_solution.h"
#include "../AGV.h"
#include "../Task.h"
#include <algorithm>
#include <limits>
#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>
#include <tuple>

#include "../TestUnit.h"

Greedy_Time_Repair::Greedy_Time_Repair(std::string s)
    : ARepairOperator(s) {
}

Greedy_Time_Repair::~Greedy_Time_Repair() {
}
void Greedy_Time_Repair::repairSolution(ISolution& sol) {
    //cout<<"Greedy_Time_Repair:"<<endl;
    AGV_solution& agv_sol = dynamic_cast<AGV_solution&>(sol);

    agv_sol.resizeVar();

    // 使用事件驱动的并行调度进行修复
    parallelEventDrivenRepair(agv_sol);

    agv_sol.calculateObjectiveValue();

    


}

void Greedy_Time_Repair::parallelEventDrivenRepair(AGV_solution& agv_sol) {
    // 初始化事件驱动变量
    std::vector<double> agv_current_time(agv_sol.pm->V, 0.0);
    std::vector<double> agv_current_soc(agv_sol.pm->V);
    std::vector<int> agv_current_task(agv_sol.pm->V, 0);

    // 初始化每辆卡车的初始电量
    for (int v = 0; v < agv_sol.pm->V; v++) {
        agv_current_soc[v] = agv_sol.agvs[v]->initial_soc;
    }
    for (int v = 0; v < agv_sol.pm->V; v++) {
        auto& agv = agv_sol.agvs[v];
        agv_current_soc[v] = agv->initial_soc;


        int num_tasks = agv_sol.pm->J_v[v];
        if (num_tasks > 0) {
            agv->task_start_times.resize(num_tasks, 0.0);
            agv->task_end_times.resize(num_tasks, 0.0);
            agv->isCharge.resize(num_tasks, false);
            agv->soc_after_task.resize(num_tasks, 0.0);
            agv->soc_before_task.resize(num_tasks, 0.0);
            agv->charging_start_times.resize(num_tasks, 0.0);
            agv->charging_end_times.resize(num_tasks, 0.0);
            agv->soc_at_cs_arrival.resize(num_tasks, 0.0);
            agv->arrival_at_station.resize(num_tasks, 0.0);
            agv->charging_durations.resize(num_tasks, 0.0);
            agv->soc_charging_durations.resize(num_tasks, 0.0);
            agv->charging_sessions.resize(num_tasks, {-1, -1});
        }
    }



    // 事件驱动的并行调度
    while (true) {
        // 找到下一个要处理的事件（最早完成当前任务的卡车）

        int next_agv = findNextTruck(agv_sol, agv_current_time, agv_current_task);

        if (next_agv == -1) break; // 所有任务完成

        // 处理选中卡车的当前任务
        processTaskWithTimeOptimalCharging(agv_sol, next_agv, agv_current_task[next_agv],
                                         agv_current_time, agv_current_soc, agv_current_task);

    }

}


int Greedy_Time_Repair::findNextTruck(AGV_solution& agv_sol,
                                               const std::vector<double>& agv_current_time,
                                               const std::vector<int>& agv_current_task) {
    int next_agv = -1;
    double earliest_time = std::numeric_limits<double>::max();

    // 遍历所有卡车，找到当前时间最早且还有未完成任务的卡车
    for (int v = 0; v < agv_sol.pm->V; v++) {
        // 检查该卡车是否还有未完成的任务
        if (agv_current_task[v] < agv_sol.agvs[v]->task_sequence.size()) {
            // 如果这辆卡车当前时间更早，则选择它
            if (agv_current_time[v] < earliest_time) {
                earliest_time = agv_current_time[v];
                next_agv = v;
            }
        }
    }

    return next_agv; // 如果所有卡车都完成了任务，返回-1
}

void Greedy_Time_Repair::processTaskWithTimeOptimalCharging(
    AGV_solution& agv_sol, int agv_id, int task_idx,
    std::vector<double>& agv_current_time,
    std::vector<double>& agv_current_soc,
    std::vector<int>& agv_current_task) {

    auto& agv = agv_sol.agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    // 1. 计算到达和执行当前任务
    double arrival_time, task_end_time , energy_consumption;

    calculateTaskTiming(agv_sol, agv_id, task_idx, agv_current_time[agv_id],
                       arrival_time, task_end_time, energy_consumption);

    // 2. 更新电量状态
    agv_current_soc[agv_id] -= energy_consumption;
    agv->soc_before_task[task_idx] = agv_current_soc[agv_id];

    agv_current_soc[agv_id] -= (current_task->cl);
    agv->soc_after_task[task_idx] = agv_current_soc[agv_id];

    agv->task_start_times[task_idx] = arrival_time;
    agv->task_end_times[task_idx] = task_end_time;
    agv_current_time[agv_id] = task_end_time;

    // if (agv->soc_before_task[task_idx] < agv_sol.pm->gamma-1e-3) {
    //
    //     cout<<"error1"<<endl;
    //     system("pause");
    // }
    // if (agv->soc_after_task[task_idx] < agv_sol.pm->gamma-1e-3) {
    //
    //     cout<<"error2"<<endl;
    //     system("pause");
    // }

    // 3. 充电决策逻辑
    bool need_charging = false;
    int selected_station = -1;

    // 检查是否是破坏算子保留的充电操作
    if (agv->isCharge[task_idx]||task_idx==agv_sol.pm->J_v[agv_id]-1) {
        if (agv->isCharge[task_idx]) {
            // 保留的充电操作，使用指定的充电站
            need_charging = true;
            selected_station = agv->charging_sessions[task_idx].first;
            // if (selected_station==-1) {
            //     cout<<"error3"<<endl;
            //     system("pause");
            // }

            // 验证保留的充电站是否可行
            double energy_to_station = current_task->cr[selected_station];

            // if (agv->soc_after_task[task_idx] - energy_to_station < agv_sol.pm->gamma) {
            //     cout<<"error4"<<endl;
            //     system("pause");
            // }
            agv->arrival_at_station[task_idx] = arrival_time + current_task->tr[selected_station];
            agv->charging_start_times[task_idx] = std::max(agv->arrival_at_station[task_idx],
                                                          agv_sol.cs_available_time[selected_station]);
            agv->charging_end_times[task_idx] = agv->charging_start_times[task_idx]+
                                                agv->charging_durations[task_idx];

            agv_current_time[agv_id] = agv->charging_end_times[task_idx];
            agv_current_soc[agv_id] = agv->soc_after_charging[task_idx];
            agv_sol.cs_available_time[selected_station] = agv->charging_end_times[task_idx];


        }



    } else {
        // 判断是否必须充电
        bool must_charge = mustChargeAfterTask(agv_sol, agv_id, task_idx, agv->soc_after_task[task_idx]);

        if (must_charge) {
            need_charging = true;
            // 修复：传入任务执行后的电量
            selected_station = selectTimeOptimalStation(agv_sol, agv_id, task_idx,
                                                      agv->soc_after_task[task_idx], task_end_time);
        }
    }

    // 4. 执行充电操作
    if (need_charging && selected_station != -1) {
        executeChargingOperation(agv_sol, agv_id, task_idx, selected_station,
                               task_end_time, agv_current_time, agv_current_soc);
        agv_current_soc[agv_id] = agv->soc_after_charging[task_idx];
    } else {
        agv_current_time[agv_id] = task_end_time;
        agv_current_soc[agv_id] = agv->soc_after_task[task_idx];
    }

    // 5. 移动到下一个任务
    agv_current_task[agv_id]++;
}

int Greedy_Time_Repair::selectTimeOptimalStation(AGV_solution& agv_sol,
                                                          int agv_id, int task_idx,
                                                          double current_soc, double task_end_time) {
    auto& agv = agv_sol.agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    int best_station = -1;
    double min_total_time = std::numeric_limits<double>::max();

    std::vector<int> feasible_stations;

    for (int c = 0; c < agv_sol.pm->C; c++) {
        // 检查电量是否足够到达充电站
        double energy_to_station = current_task->cr[c];
        double soc_at_station = current_soc - energy_to_station;

        if (soc_at_station < agv_sol.pm->gamma) {
            continue; // 电量不足，跳过此充电站
        }

        feasible_stations.push_back(c);

        // 计算总时间
        double total_time = calculateTotalChargingTime(agv_sol, agv_id, task_idx, c,
                                                      task_end_time, current_soc);

        if (total_time < min_total_time) {
            min_total_time = total_time;
            best_station = c;
        }
    }

    // 如果没有找到可行的充电站，输出详细调试信息
    if (best_station == -1)
    {
        //cout<<"error4"<<endl;
        //tem("pause");
        // 如果确实没有可行站点，返回-1，让上层处理
        return 0;
    }


    return best_station;
}

void Greedy_Time_Repair::calculateTaskTiming(AGV_solution& agv_sol,
                                                      int agv_id, int task_idx,
                                                      double current_time,
                                                      double& arrival_time,
                                                      double& task_end_time,
                                                      double& energy_consumption) {
    auto& agv = agv_sol.agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    // 计算到达当前任务起点的时间和能耗
    double travel_time_to_task = 0.0;

    if (task_idx == 0) {
        // 第一个任务，从起始位置出发
        travel_time_to_task = current_task->te;
        energy_consumption = current_task->ce;
    } else {
        // 不是第一个任务，需要考虑前一个任务的情况
        if (agv->isCharge[task_idx - 1]) {
            // 前一个任务有充电，从充电站出发
            int charging_station = agv->charging_sessions[task_idx - 1].first;
            travel_time_to_task = current_task->tf[charging_station];
            energy_consumption = current_task->cf[charging_station];
        } else {
            // 前一个任务没有充电，从前一个任务的结束位置出发
            travel_time_to_task = current_task->te;
            energy_consumption = current_task->ce;
        }
    }

    // 计算时间
    arrival_time = current_time + travel_time_to_task;
    task_end_time = arrival_time + current_task->tl;
}

void Greedy_Time_Repair::executeChargingOperation(AGV_solution& agv_sol,
                                                           int agv_id, int task_idx,
                                                           int station_id, double task_end_time,
                                                           std::vector<double>& agv_current_time,
                                                           std::vector<double>& agv_current_soc) {
    auto& agv = agv_sol.agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    // 1. 设置充电标志和充电站
    agv->isCharge[task_idx] = true;
    agv->charging_sessions[task_idx] = {station_id, -1};

    

    // 3. 计算充电时间安排
    double arrival_at_cs = task_end_time + current_task->tr[station_id];
    double charging_start_time = std::max(arrival_at_cs, agv_sol.cs_available_time[station_id]);
    agv->soc_at_cs_arrival[task_idx] = agv->soc_after_task[task_idx]-agv->task_sequence[task_idx]->cr[station_id];
    agv->isCharge[task_idx] = true;
    agv->charging_sessions[task_idx] = {station_id, -1};
    // 查找下一个充电点
    int next_charge_pos = -1;
    for (int i = task_idx + 1; i < agv->task_sequence.size(); ++i) {
        if (agv->isCharge[i]) {
            next_charge_pos = i;
            break;
        }
    }

    if (next_charge_pos != -1) {
        // 1. 先假设充满电，模拟能否安全到达下一个充电点
        double max_charge = agv_sol.pm->Pi - agv->soc_at_cs_arrival[task_idx];
        double cur_soc = agv->soc_at_cs_arrival[task_idx] + max_charge;
        int prev_station = station_id;
        bool feasible = true;
        for (int i = task_idx + 1; i <= next_charge_pos; ++i) {
            Task* t = agv->task_sequence[i];
            if (i == task_idx + 1) {
                cur_soc -= t->cf[prev_station];
            } else {
                cur_soc -= t->ce;
            }
            cur_soc -= t->cl;
            if (i == next_charge_pos) {
                int next_station = agv->charging_sessions[next_charge_pos].first;
                cur_soc -= t->cr[next_station];
            }
            if (cur_soc < agv_sol.pm->gamma-1e-3) {
                feasible = false;
                break;
            }
        }
        if (!feasible) {
            // 只能充满
            agv->soc_charging_durations[task_idx] = max_charge;
            agv->soc_after_charging[task_idx] = agv->soc_at_cs_arrival[task_idx] + max_charge;

            agv->charging_durations[task_idx] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx] , agv->soc_after_charging[task_idx]) + agv_sol.pm->tu;
//            if (agv_sol.pm->tm->warm_start) {
//                agv->charging_durations[task_idx] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx] , agv->soc_after_charging[task_idx]) + agv_sol.pm->tu;
//
//            }
//            else {
//                agv->charging_durations[task_idx] = (agv->soc_after_charging[task_idx]-agv->soc_at_cs_arrival[task_idx])/agv_sol.pm->eta + agv_sol.pm->tu;
//            }
        } else {
            // 可以精确充，计算最优充电量
            double total_consume = 0.0;
            prev_station = station_id;
            for (int i = task_idx + 1; i <= next_charge_pos; ++i) {
                Task* t = agv->task_sequence[i];
                if (i == task_idx + 1) {
                    total_consume += t->cf[prev_station];
                } else {
                    total_consume += t->ce;
                }
                total_consume += t->cl;
                if (i == next_charge_pos) {
                    int next_station = agv->charging_sessions[next_charge_pos].first;
                    total_consume += t->cr[next_station];
                }
            }
            double required_charge = agv->soc_at_cs_arrival[next_charge_pos] + total_consume - agv->soc_at_cs_arrival[task_idx];
            //if (required_charge < 0) required_charge = 0;

            if (required_charge > max_charge) required_charge = max_charge;
            agv->soc_charging_durations[task_idx] = required_charge;
            agv->soc_after_charging[task_idx] = agv->soc_at_cs_arrival[task_idx] + required_charge;

            agv->charging_durations[task_idx] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx], agv->soc_after_charging[task_idx]) + agv_sol.pm->tu;
//            if (agv_sol.pm->tm->warm_start) {
//                agv->charging_durations[task_idx] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx] , agv->soc_after_charging[task_idx]) + agv_sol.pm->tu;
//
//            }
//            else {
//                agv->charging_durations[task_idx] = (agv->soc_after_charging[task_idx]-agv->soc_at_cs_arrival[task_idx])/agv_sol.pm->eta + agv_sol.pm->tu;
//            }
        }
    }
    //表明后续已经没有
    else {
        // 1. 假设充满电，模拟能否完成所有剩余任务且最后剩余电量≥gamma
        double max_charge = agv_sol.pm->Pi - agv->soc_at_cs_arrival[task_idx];
        double cur_soc = agv->soc_at_cs_arrival[task_idx] + max_charge;
        int prev_station = station_id;
        bool feasible = true;
        for (int i = task_idx + 1; i < agv->task_sequence.size(); ++i) {
            Task* t = agv->task_sequence[i];
            if (i == task_idx + 1) {
                cur_soc -= t->cf[prev_station];
            } else if (agv->isCharge[i - 1]) {
                int prev_cs = agv->charging_sessions[i - 1].first;
                cur_soc -= t->cf[prev_cs];
            } else {
                cur_soc -= t->ce;
            }
            cur_soc -= t->cl;
            // 没有后续充电站，不需要再减去cr
            if (cur_soc < agv_sol.pm->gamma-1e-3) {
                feasible = false;
                break;
            }
        }
        // 2. 判断最后剩余电量
        if (feasible && cur_soc >= agv_sol.pm->gamma-1e-3) {
            // 可以精确充，计算最优充电量
            double total_consume = 0.0;
            prev_station = station_id;
            for (int i = task_idx + 1; i < agv->task_sequence.size(); ++i) {
                Task* t = agv->task_sequence[i];
                if (i == task_idx + 1) {
                    total_consume += t->cf[prev_station];
                } else if (agv->isCharge[i - 1]) {
                    int prev_cs = agv->charging_sessions[i - 1].first;
                    total_consume += t->cf[prev_cs];
                } else {
                    total_consume += t->ce;
                }
                total_consume += t->cl;
            }
            double required_charge = agv_sol.pm->gamma + total_consume - agv->soc_at_cs_arrival[task_idx];
            //if (required_charge < 0) required_charge = 0;

            if (required_charge > max_charge) required_charge = max_charge;
            agv->soc_charging_durations[task_idx] = required_charge;
            agv->soc_after_charging[task_idx] = agv->soc_at_cs_arrival[task_idx] + required_charge;

            agv->charging_durations[task_idx] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx], agv->soc_after_charging[task_idx]) + agv_sol.pm->tu;

//            if (agv_sol.pm->tm->warm_start) {
//                agv->charging_durations[task_idx] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx] , agv->soc_after_charging[task_idx]) + agv_sol.pm->tu;
//
//            }
//            else {
//                agv->charging_durations[task_idx] = (agv->soc_after_charging[task_idx]-agv->soc_at_cs_arrival[task_idx])/agv_sol.pm->eta + agv_sol.pm->tu;
//            }
        } else {
            // 只能充满
            agv->soc_charging_durations[task_idx] = max_charge;
            agv->soc_after_charging[task_idx] = agv->soc_at_cs_arrival[task_idx] + max_charge;

            agv->charging_durations[task_idx] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx], agv->soc_after_charging[task_idx]) + agv_sol.pm->tu;
//            if (agv_sol.pm->tm->warm_start) {
//                agv->charging_durations[task_idx] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx] , agv->soc_after_charging[task_idx]) + agv_sol.pm->tu;
//
//            }
//            else {
//                agv->charging_durations[task_idx] = (agv->soc_after_charging[task_idx]-agv->soc_at_cs_arrival[task_idx])/agv_sol.pm->eta + agv_sol.pm->tu;
//            }
        }
    }

    
    double charging_end_time = charging_start_time + agv->charging_durations[task_idx];

    // 4. 记录充电相关信息
    agv->arrival_at_station[task_idx] = arrival_at_cs;
    agv->charging_start_times[task_idx] = charging_start_time;
    agv->charging_end_times[task_idx] = charging_end_time;

    // 修复：使用任务执行后的电量（agv->soc_after_task[task_idx]）减去到充电站的能耗
    agv->soc_at_cs_arrival[task_idx] = agv->soc_after_task[task_idx] - current_task->cr[station_id];

    // if (agv->soc_at_cs_arrival[task_idx] < agv_sol.pm->gamma-1e-3) {
    //     cout<<"error6"<<endl;
    //     system("pause");
    // }

    // 5. 更新统计信息

    agv_sol.total_charging_sessions++;
    agv_sol.total_charging_time += agv->charging_durations[task_idx];

    double waiting_time = charging_start_time - arrival_at_cs;
    if (waiting_time > 0) {
        agv_sol.total_waiting_time += waiting_time;
    }

    // 6. 更新充电站可用时间
    agv_sol.cs_available_time[station_id] = charging_end_time;

    // 7. 更新卡车状态
    agv_current_soc[agv_id] = agv->soc_after_charging[task_idx]; // 充电后满电
    agv_current_time[agv_id] = charging_end_time;   // 更新车辆当前时间
}

bool Greedy_Time_Repair::mustChargeAfterTask(AGV_solution& agv_sol,
                                                      int agv_id, int task_idx, double current_soc) {
    auto& agv = agv_sol.agvs[agv_id];
    auto& current_task = agv->task_sequence[task_idx];
    if (task_idx >= agv->task_sequence.size() - 1) {
        return false; // 最后一个任务后不需要充电
    }
    if (agv->isCharge[task_idx]) {
        return false;
    }
    bool need_charge = false;
    if (task_idx<agv->soc_before_task.size()-2) {
        // 计算本任务和到下一个任务/充电站的总能耗
        double next_soc = agv_sol.simulateTaskExecution(agv_id, task_idx+1, current_soc);

        next_soc -= (agv->task_sequence[task_idx + 1]->cl);
        if (agv->isCharge[task_idx+1]) {
            double min_soc_after_next = next_soc;
            double min_soc_to_station = agv->task_sequence[task_idx+1]->cr[agv->charging_sessions[task_idx + 1].first];
            min_soc_after_next = min_soc_after_next - min_soc_to_station;
            if (min_soc_after_next < agv_sol.pm->gamma-1e-3) {
                need_charge = true;
            }
        }
        else {
            // 预判：本任务+到下一个任务/充电站的能耗，是否全程大于 gamma
            double min_soc_after_next = next_soc;
            double min_soc_to_station = std::numeric_limits<double>::max();
            for (int c = 0; c < agv_sol.pm->C; c++) {
                double soc_to_station = agv->task_sequence[task_idx + 1]->cr[c];
                if (soc_to_station < min_soc_to_station) {
                    min_soc_to_station = soc_to_station;
                }
            }
            min_soc_after_next = min_soc_after_next - min_soc_to_station;
            if (min_soc_after_next < agv_sol.pm->gamma-1e-3) {
                need_charge = true;

            }
        }
    }
    else {
        // 计算本任务和到下一个任务/充电站的总能耗
        double next_soc = agv_sol.simulateTaskExecution(agv_id, task_idx+1, current_soc);
        next_soc -= (agv->task_sequence[task_idx + 1]->cl);
        // 预判：本任务+到下一个任务/充电站的能耗，是否全程大于 gamma
        double min_soc_after_next = next_soc;
        if (min_soc_after_next < agv_sol.pm->gamma-1e-3) {
            need_charge = true;
        }
    }

    return need_charge;
}

//没有考虑充电的时间
double Greedy_Time_Repair::calculateTotalChargingTime(AGV_solution& agv_sol,
                                                               int agv_id, int task_idx, int station_id,
                                                               double task_end_time, double current_soc) {
    auto& agv = agv_sol.agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    // 1. 前往充电站的时间
    double travel_to_station_time = current_task->tr[station_id];

    // 2. 到达充电站的时间
    double arrival_at_station = task_end_time + travel_to_station_time;

    // 3. 等待时间（排队时间）
    double charging_start_time = std::max(arrival_at_station, agv_sol.cs_available_time[station_id]);
    double waiting_time = charging_start_time - arrival_at_station;
    //只考虑了固定的充电的时间
    // 4. 充电时间
    double charging_time = agv_sol.pm->tu;

    // 5. 如果有下一个任务，计算从充电站到下一个任务取货点的时间
    double travel_to_next_time = 0.0;
    if (task_idx < agv->task_sequence.size() - 1) {
        Task* next_task = agv->task_sequence[task_idx + 1];
        travel_to_next_time = next_task->tf[station_id];
    }

    // 总时间 = 前往充电站时间 + 等待时间 + 充电时间 + 到下一任务时间
    return travel_to_station_time + waiting_time + charging_time + travel_to_next_time;
}



