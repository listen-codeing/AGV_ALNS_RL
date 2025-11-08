//
// Created by 张欢 on 2025/8/9.
//

#include "Adaptive_Best_Repair.h"
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

using namespace std;

// 添加调试标志定义


Adaptive_Best_Repair::Adaptive_Best_Repair(std::string s)
    : ARepairOperator(s) {
}

Adaptive_Best_Repair::~Adaptive_Best_Repair() {
}

void Adaptive_Best_Repair::repairSolution(ISolution& sol) {
    //cout<<"Adaptive_Best_Repair"<<endl;
    AGV_solution& agv_sol = dynamic_cast<AGV_solution&>(sol);
    agv_sol.resizeVar();


    // Phase 1: 为每辆卡车进行间隔完整性修复（确定充电位置和充电站）
    for (int v = 0; v < agv_sol.pm->V; v++) {


        processInterval(agv_sol,v);
    }

    recalculateCompleteSchedule(agv_sol);

    agv_sol.calculateObjectiveValue();




}

void Adaptive_Best_Repair::recalculateCompleteSchedule(AGV_solution& agv_sol){

    agv_sol.cs_available_time.assign(agv_sol.pm->C, 0.0);
    agv_sol.total_waiting_time = 0.0;
    agv_sol.total_charging_time = 0.0;
    agv_sol.total_charging_sessions = 0;
    agv_sol.cs.clear();


    // 初始化事件驱动变量
    std::vector<double> agv_current_time(agv_sol.pm->V, 0.0);
    std::vector<double> agv_current_soc(agv_sol.pm->V);
    std::vector<int> agv_current_task(agv_sol.pm->V, 0);

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
        // 找到下一个要处理的事件（最早的未完成任务）
        int next_agv = -1;
        double earliest_time = std::numeric_limits<double>::max();

        for (int v = 0; v < agv_sol.pm->V; v++) {
            if (agv_current_task[v] < agv_sol.agvs[v]->task_sequence.size()) {
                if (agv_current_time[v] < earliest_time) {
                    earliest_time = agv_current_time[v];
                    next_agv = v;
                }
            }
        }

        // 如果没有待处理任务，结束调度
        if (next_agv == -1) break;

        // 处理选中车辆的当前任务
        int v = next_agv;
        int j = agv_current_task[v];
        auto& agv = agv_sol.agvs[v];
        Task* current_task = agv->task_sequence[j];

        // 1. 计算到达当前任务起点的时间和能耗
        double travel_time_to_task = 0.0;
        double energy_consumption_to_task = 0.0;

        if (j == 0) {
            travel_time_to_task = current_task->te;
            energy_consumption_to_task = current_task->ce;
        } else {
            if (agv->isCharge[j-1]) {
                int charging_station = agv->charging_sessions[j-1].first;
                travel_time_to_task = current_task->tf[charging_station];
                energy_consumption_to_task = current_task->cf[charging_station];
            } else {
                travel_time_to_task = current_task->te;
                energy_consumption_to_task = current_task->ce;
            }
        }

        // 2. 计算任务开始和结束时间
        agv->task_start_times[j] = agv_current_time[v] + travel_time_to_task;
        agv->task_end_times[j] = agv->task_start_times[j] + current_task->tl;

        // 3. 更新电量状态
        agv_current_soc[v] -= energy_consumption_to_task;

        agv->soc_before_task[j] = agv_current_soc[v];
        // if (agv->soc_before_task[j] <agv_sol.pm->gamma-1e-3) {
        //     cout<<"agv->soc_before_task[j]:"<<agv->soc_before_task[j]<<endl;
        //     system("pause");
        // }
        agv_current_soc[v] -= (current_task->cl);

        agv->soc_after_task[j] = agv_current_soc[v];
        // if (agv->soc_after_task[j] <agv_sol.pm->gamma-1e-3) {
        //     cout<<"agv->soc_after_task[j]:"<<agv->soc_after_task[j]<<endl;
        //     system("pause");
        // }

        // 4. 判断是否需要充电
        if (agv->isCharge[j]) {

            int station_id = agv->charging_sessions[j].first;
            double arrival_at_cs = agv->task_end_times[j] + current_task->tr[station_id];
            double charging_start_time = std::max(arrival_at_cs, agv_sol.cs_available_time[station_id]);
            double charging_end_time = charging_start_time + agv->charging_durations[j];
            agv->arrival_at_station[j] = arrival_at_cs;
            agv->soc_at_cs_arrival[j] = agv_current_soc[v] - current_task->cr[station_id];
            // if (agv->soc_at_cs_arrival[j]<agv_sol.pm->gamma-1e-3) {
            //     cout<<"agv->soc_at_cs_arrival[j]:"<<agv->soc_at_cs_arrival[j]<<endl;
            //     cout<<"big error3"<<endl;
            //     system("pause");
            // }


            agv->charging_start_times[j] = charging_start_time;
            agv->charging_end_times[j] = charging_end_time;
            agv_sol.cs.push_back(std::make_tuple(v, j, charging_start_time, agv->charging_durations[j]));
            agv_sol.total_charging_sessions++;
            agv_sol.total_charging_time += agv->charging_durations[j];

            double waiting_time = charging_start_time - arrival_at_cs;
            if (waiting_time > 0) {
                agv_sol.total_waiting_time += waiting_time;
            }

            agv_sol.cs_available_time[station_id] = charging_end_time;
            agv_current_soc[v] =agv->soc_after_charging[j];
            agv_current_time[v] = charging_end_time; // 更新车辆当前时间
        } else {
            agv_current_time[v] = agv->task_end_times[j]; // 更新车辆当前时间
        }

        // 移动到下一个任务
        agv_current_task[v]++;
    }

    // 计算makespan
    agv_sol.makespan = 0.0;
    //
    for (int v = 0; v < agv_sol.pm->V; v++) {
        if (!agv_sol.agvs[v]->task_sequence.empty()) {
            agv_sol.makespan = std::max(agv_sol.makespan, agv_current_time[v]);
        }
    }
    agv_sol.calculateObjectiveValue();


}


void Adaptive_Best_Repair::processInterval(AGV_solution& agv_sol, int v) {
    auto& agv = agv_sol.agvs[v];
    //最低电量
    double gamma = agv_sol.pm->gamma;
    //满电
    double Pi = agv_sol.pm->Pi;
    int num_tasks = agv->task_sequence.size();
    int last_charge_idx = -1;
    double current_soc = agv->initial_soc;
    Task* current_task = agv->task_sequence[0];



    for (int pos = 0; pos < agv->task_sequence.size()-1; pos++) {

        current_task = agv->task_sequence[pos];
        current_soc = agv_sol.simulateTaskExecution(v, pos, current_soc);
        agv->soc_before_task[pos] = current_soc;
        current_soc -= (current_task->cl);
        agv->soc_after_task[pos] = current_soc;

        bool need_charge = false;
        if (agv->isCharge[pos]) {
            int station_id = agv->charging_sessions[pos].first;
            current_soc -= current_task->cr[station_id];
            agv->soc_at_cs_arrival[pos] = current_soc;

            // if (agv->soc_at_cs_arrival[pos] < gamma-1e-3) {
            //     cout << "here2 error" << endl;
            //     system("pause");
            // }
            current_soc = agv->soc_after_charging[pos];

        }
        else {
            if (pos<agv->soc_before_task.size()-2) {
                // 计算本任务和到下一个任务/充电站的总能耗

                double next_soc = agv_sol.simulateTaskExecution(v, pos+1, current_soc);

                next_soc -= (agv->task_sequence[pos + 1]->cl);

                if (agv->isCharge[pos+1]) {
                    double min_soc_after_next = next_soc;
                    double min_soc_to_station = agv->task_sequence[pos+1]->cr[agv->charging_sessions[pos + 1].first];

                    min_soc_after_next = min_soc_after_next - min_soc_to_station;

                    if (min_soc_after_next < agv->soc_at_cs_arrival[pos+1]) {



                        need_charge = true;
                    }
                }
                else {
                    // 预判：本任务+到下一个任务/充电站的能耗，是否全程大于 gamma
                    double min_soc_after_next = next_soc;
                    double min_soc_to_station = std::numeric_limits<double>::max();
                    for (int c = 0; c < agv_sol.pm->C; c++) {
                        double soc_to_station = agv->task_sequence[pos + 1]->cr[c];
                        if (soc_to_station < min_soc_to_station) {
                            min_soc_to_station = soc_to_station;
                        }
                    }

                    min_soc_after_next = min_soc_after_next - min_soc_to_station;

                    if (min_soc_after_next < gamma-1e-3) {


                        need_charge = true;

                    }
                }
            }
            else {
                // 计算本任务和到下一个任务/充电站的总能耗
                double next_soc = agv_sol.simulateTaskExecution(v, pos+1, current_soc);
                next_soc -= (agv->task_sequence[pos + 1]->cl);
                // 预判：本任务+到下一个任务/充电站的能耗，是否全程大于 gamma
                double min_soc_after_next = next_soc;

                if (min_soc_after_next < gamma-1e-3) {

                    need_charge = true;
                }
            }

        }
        if (need_charge && !agv->isCharge[pos]) {

            int prev_charge_idx = -1;
            for (int i = pos - 1; i >= 0; --i) {
                if (agv->isCharge[i]) {
                    prev_charge_idx = i;
                    break;
                }
            }
            prev_charge_idx+=1;
            static std::mt19937 rng(std::random_device{}());

            std::vector<int> weights(pos - prev_charge_idx + 1);
            for (int i = 0; i < weights.size(); ++i) {
                weights[i] = i + 1; // 权重越靠后越大
            }

            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            int insert_pos = prev_charge_idx + dist(rng);

            pos = insert_pos;


            int station_id = selectOptimalChargingStation(agv_sol, v, pos, agv->soc_after_task[pos]);

            addChargingOperation(agv_sol,v,pos,station_id);

            agv->soc_at_cs_arrival[pos] = agv->soc_after_task[pos]-agv->task_sequence[pos]->cr[station_id];

            // if (agv->soc_at_cs_arrival[pos]<gamma-1e-3) {
            //     cout<<"energy error1"<<endl;
            //     system("pause");
            // }
            agv->isCharge[pos] = true;
            agv->charging_sessions[pos] = {station_id, -1};

            // 查找下一个充电点
            int next_charge_pos = -1;
            for (int i = pos + 1; i < agv->task_sequence.size()-1; ++i) {
                if (agv->isCharge[i]) {
                    next_charge_pos = i;
                    break;
                }
            }

            if (next_charge_pos != -1) {

                // 1. 先假设充满电，模拟能否安全到达下一个充电点
                double max_charge = Pi - agv->soc_at_cs_arrival[pos];
                double cur_soc = agv->soc_at_cs_arrival[pos] + max_charge;
                int prev_station = station_id;
                bool feasible = true;

                for (int i = pos + 1; i <= next_charge_pos; ++i) {
                    Task* t = agv->task_sequence[i];
                    if (i == pos + 1) {
                        cur_soc -= t->cf[prev_station];
                    } else {
                        cur_soc -= t->ce;
                    }
                    cur_soc -= t->cl;

                    if (i == next_charge_pos) {
                        int next_station = agv->charging_sessions[next_charge_pos].first;
                        cur_soc -= t->cr[next_station];

                        // 关键：这里应该检查是否能达到下一个充电点的预期电量
                        if (cur_soc < agv->soc_at_cs_arrival[next_charge_pos]) {
                            feasible = false;
                            break;
                        }
                    } else {
                        // 中间任务只需要保证电量不低于gamma
                        if (cur_soc < gamma-1e-3) {
                            feasible = false;
                            break;
                        }
                    }
                }
                if (feasible) {

                    // 可以精确充，计算最优充电量
                    double total_consume = 0.0;
                    prev_station = station_id;
                    for (int i = pos + 1; i <= next_charge_pos; ++i) {
                        Task* t = agv->task_sequence[i];
                        if (i == pos + 1) {
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
                    double required_charge = agv->soc_at_cs_arrival[next_charge_pos] + total_consume - agv->soc_at_cs_arrival[pos];

                    //if (required_charge < 0) required_charge = 0;

                    if (required_charge > max_charge) required_charge = max_charge;

                    agv->soc_charging_durations[pos] = required_charge;
                    agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + required_charge;

                    agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                    //利用warm_start判断是否是线性充电
//                    if (agv_sol.pm->tm->warm_start) {
//                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                    }
//                    else {
//                        agv->charging_durations[pos] = (agv->soc_after_charging[pos]-agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta+agv_sol.pm->tu;
//                    }

                } else {
                    // 计算最小充电量

                    Task* next_task = agv->task_sequence[pos + 1];
                    double min_soc_to_next_cs = std::numeric_limits<double>::max();
                    if (agv->isCharge[pos + 1]) {
                        int station = agv->charging_sessions[pos + 1].first;
                        min_soc_to_next_cs = agv->task_sequence[pos+1]->cr[station];
                    } else {
                        for (int c = 0; c < agv_sol.pm->C; c++) {
                            double soc_to_station = agv->task_sequence[pos+1]->cr[c];
                            if (soc_to_station < min_soc_to_next_cs) {
                                min_soc_to_next_cs = soc_to_station;
                            }
                        }
                    }

                    double min_charge = next_task->cf[station_id] + next_task->cl + min_soc_to_next_cs + gamma - agv->soc_at_cs_arrival[pos];

                    if (min_charge < 0) min_charge =0;
                    if (min_charge > max_charge) min_charge = max_charge;
                    if (max_charge-min_charge<1) {
                        agv->soc_charging_durations[pos] = max_charge;
                        agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + max_charge;
                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos]) + agv_sol.pm->tu;
//                        if (agv_sol.pm->tm->warm_start) {
//                            agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos]) + agv_sol.pm->tu;
//                        }
//                        else {
//                            agv->charging_durations[pos] = (agv->soc_after_charging[pos]-agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta+agv_sol.pm->tu;
//                        }
                    }
                    else {
                        int n = 20; // 分段数量
                        double step = (max_charge - min_charge) / n; // 计算步长
                        std::vector<double> candidates;
                        // 从 min_charge + step 开始生成候选值
                        for (int i = 1; i <= n; ++i) {
                            candidates.push_back(min_charge + i * step);
                        }
                        // 如果没有候选值，兜底使用 min_charge + step
                        if (candidates.empty()) {
                            candidates.push_back(min_charge + step);
                        }

                        static std::mt19937 rng(std::random_device{}());
                        std::uniform_int_distribution<int> idx_dist(0, candidates.size() - 1);
                        double random_charge = candidates[idx_dist(rng)];

                        agv->soc_charging_durations[pos] = random_charge;
                        agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + random_charge;

                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos]) + agv_sol.pm->tu;
//                        if (agv_sol.pm->tm->warm_start) {
//                            agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos]) + agv_sol.pm->tu;
//                        }
//                        else {
//                            agv->charging_durations[pos] = (agv->soc_after_charging[pos]-agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta+agv_sol.pm->tu;
//                        }
                    }

                }
            }
            //表明后续已经没有
            else {

                // 1. 假设充满电，模拟能否完成所有剩余任务且最后剩余电量≥gamma
                double max_charge = Pi - agv->soc_at_cs_arrival[pos];
                double cur_soc = agv->soc_at_cs_arrival[pos] + max_charge;
                int prev_station = station_id;
                bool feasible = true;
                for (int i = pos + 1; i < agv->task_sequence.size(); ++i) {
                    Task* t = agv->task_sequence[i];
                    if (i == pos + 1) {
                        cur_soc -= t->cf[prev_station];
                    } else if (agv->isCharge[i - 1]) {
                        int prev_cs = agv->charging_sessions[i - 1].first;
                        cur_soc -= t->cf[prev_cs];
                    } else {
                        cur_soc -= t->ce;
                    }
                    cur_soc -= t->cl;
                    // 没有后续充电站，不需要再减去cr
                    if (cur_soc < gamma-1e-3) {
                        feasible = false;
                        break;
                    }
                }

                // 2. 判断最后剩余电量
                if (feasible && cur_soc >= gamma) {

                    // 可以精确充，计算最优充电量
                    double total_consume = 0.0;
                    prev_station = station_id;
                    for (int i = pos + 1; i < agv->task_sequence.size(); ++i) {
                        Task* t = agv->task_sequence[i];
                        if (i == pos + 1) {
                            total_consume += t->cf[prev_station];
                        } else if (agv->isCharge[i - 1]) {
                            int prev_cs = agv->charging_sessions[i - 1].first;
                            total_consume += t->cf[prev_cs];
                        } else {
                            total_consume += t->ce;
                        }
                        total_consume += t->cl;
                    }

                    double required_charge = gamma + total_consume - agv->soc_at_cs_arrival[pos];

                    //if (required_charge < 0) required_charge = 0;
                    // if (required_charge < 0) {
                    //     agv->isCharge[pos] = false;
                    //     agv->charging_sessions[pos] = {-1, -1};
                    //     continue;
                    //
                    // }
                    if (required_charge > max_charge) required_charge = max_charge;


                    agv->soc_charging_durations[pos] = required_charge;
                    agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + required_charge;

                    agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                    if (agv_sol.pm->tm->warm_start) {
//                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                    }
//                    else {
//                        agv->charging_durations[pos] = (agv->soc_after_charging[pos]-agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta+agv_sol.pm->tu;
//                    }
                    //后续可以不用遍历了，所有的操作都结束了，
                    //break;
                } else {

                    // 计算最小充电量
                    Task* next_task = agv->task_sequence[pos + 1];

                    double min_soc_to_next_cs = std::numeric_limits<double>::max();

                    if (agv->isCharge[pos + 1]) {

                        min_soc_to_next_cs = next_task->cr[agv->charging_sessions[pos + 1].first];
                    } else {
                        for (int c = 0; c < agv_sol.pm->C; c++) {

                            double soc_to_station = agv->task_sequence[pos + 1]->cr[c];
                            if (soc_to_station < min_soc_to_next_cs) {
                                min_soc_to_next_cs = soc_to_station;
                            }
                        }
                    }


                    double min_charge = next_task->cf[station_id] + next_task->cl + min_soc_to_next_cs + gamma - agv->soc_at_cs_arrival[pos];


                    if (min_charge < 0) min_charge = 0;
                    if (min_charge > max_charge) min_charge = max_charge;
                    if (max_charge - min_charge<1) {
                        agv->soc_charging_durations[pos] = max_charge;
                        agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + max_charge;
                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                        if (agv_sol.pm->tm->warm_start) {
//                            agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//
//                        }
//                        else {
//                            agv->charging_durations[pos] = (agv->soc_after_charging[pos]-agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta+agv_sol.pm->tu;
//                        }
                    }
                    else {
                        int n = 20; // 分段数量
                        double step = (max_charge - min_charge) / n; // 计算步长
                        std::vector<double> candidates;

                        // 从 min_charge + step 开始生成候选值
                        for (int i = 1; i <= n; ++i) {
                            candidates.push_back(min_charge + i * step);
                        }

                        // 如果没有候选值，兜底使用 min_charge + step
                        if (candidates.empty()) {
                            candidates.push_back(min_charge + step);
                        }

                        static std::mt19937 rng(std::random_device{}());
                        std::uniform_int_distribution<int> idx_dist(0, candidates.size() - 1);
                        double random_charge = candidates[idx_dist(rng)];

                        agv->soc_charging_durations[pos] = random_charge;
                        agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + random_charge;
                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                        if (agv_sol.pm->tm->warm_start) {
//                            agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                        }
//                        else {
//                            agv->charging_durations[pos] = (agv->soc_after_charging[pos]-agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta+agv_sol.pm->tu;
//                        }
                    }

                }
            }

        }
    }
}

int Adaptive_Best_Repair::selectOptimalChargingStation(AGV_solution& agv_sol, int agv_id, int pos, double current_soc) {
    // if (agv_sol.pm->C == 1) {
    //     return 0;
    // }

    auto& agv = agv_sol.agvs[agv_id];
    Task* current_task = agv->task_sequence[pos];
    int num_tasks = agv->task_sequence.size();

    std::vector<int> feasible_stations;

    // 筛选可行充电站
    for (int c = 0; c < agv_sol.pm->C; c++) {
        double energy_to_station = current_task->cr[c];
        double soc_after_charging = current_soc - energy_to_station;
        if (soc_after_charging >= agv_sol.pm->gamma-1e-3) {
            feasible_stations.push_back(c);
        }
    }

    if (feasible_stations.empty()) {
        cout << "Warning: No feasible charging stations found, using station 0" << endl;
        //system("pause");
        return 0;
    }

    // 选择总时间最小的充电站
    int best_station = feasible_stations[0];
    double min_time = std::numeric_limits<double>::max();

    for (int c : feasible_stations) {
        double time_to_station = current_task->tr[c];
        double time_from_station_to_next = 0.0;
        if (pos < num_tasks - 1) {
            time_from_station_to_next = current_task->tf[c];
        }
        double total_time = time_to_station + time_from_station_to_next;
        if (total_time < min_time) {
            min_time = total_time;
            best_station = c;
        }
    }

    return best_station;
}



void Adaptive_Best_Repair::addChargingOperation(AGV_solution& agv_sol,int agv_id, int pos, int station_id) {
    auto& agv = agv_sol.agvs[agv_id];

    // 设置充电决策
    agv->isCharge[pos] = true;
    agv->charging_sessions[pos] = {station_id, -1};

}


