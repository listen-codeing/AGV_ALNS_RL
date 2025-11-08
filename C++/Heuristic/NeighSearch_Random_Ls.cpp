//
// Created by 张欢 on 2025/7/31.
//

#include "NeighSearch_Random_Ls.h"

#include "AGV_solution.h"
#include "../AGV.h"
#include<random>
#include<algorithm>


//todo

NeighSearch_Random_Ls::~NeighSearch_Random_Ls() {

}

NeighSearch_Random_Ls::NeighSearch_Random_Ls(std::string s)
{
    name = s;
}

bool NeighSearch_Random_Ls::performLocalSearch(ISolution &sol) {
    //cout<<"performLocalSearch4:"<<endl;
    AGV_solution& agv_sol = dynamic_cast<AGV_solution&>(sol);
    double perCost = agv_sol.getObjectiveValue();
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::pair<int, int>> max_waiting_indices;
    std::vector<std::vector<std::pair<int, int>>> total_waiting_indices;
    size_t max_size = 2;
    int selectStaion = -1;
    double window_before = 30; // 向前10时间单位
    double window_after = 30;  // 向后10时间单位
    // 1. 找到最大排队集合
    for (int v = 0; v < agv_sol.pm->V; v++) {
        auto& agv = agv_sol.agvs[v];
        for (int j = 0; j < agv_sol.pm->J_v[v]; j++) {
            Task* current_task = agv->task_sequence[j];

            if (agv->isCharge[j]) {

                int cs_id = agv->charging_sessions[j].first;

                double arrive_time = agv->arrival_at_station[cs_id];

                std::vector<std::pair<int, int>> waiting_indices;

                for (int vv = 0; vv < agv_sol.pm->V; vv++) {
                    auto& t_agv = agv_sol.agvs[vv];
                    for (int jj = 0; jj < t_agv->charging_end_times.size(); jj++) {
                        Task* t = t_agv->task_sequence[jj];
                        if (t->isCharging &&
                            t_agv->charging_sessions[jj].first == cs_id) {
                            double charging_start = t_agv->charging_start_times[jj];
                            if (charging_start >= arrive_time - window_before &&
                                charging_start <= arrive_time + window_after &&vv!=v) {
                                waiting_indices.emplace_back(vv, jj);
                                }
                            }
                    }
                }

                if (waiting_indices.size() > 1) {
                    total_waiting_indices.push_back(waiting_indices);
                }

            }
        }
    }

    int temp = total_waiting_indices.size();
    if (temp ==0) {
        return false;
    }
    int rand_idx  =0;
    if (temp > 0) {
        std::uniform_int_distribution<> dis(0, temp - 1);
        rand_idx = dis(gen);
        // rand_idx 就是你要的随机下标
    }
    max_waiting_indices = total_waiting_indices[rand_idx];

        // 2. 局部打乱最大集合并更新Y
    double current_time =0;
    if (max_waiting_indices.size() > 1) {

        double cs_available_time = 0.0;
        for (const auto& idx : max_waiting_indices) {
            int v = idx.first;
            int j = idx.second;
            auto& agv = agv_sol.agvs[v];
            int cs_id = agv->charging_sessions[j].first;
            double arrival_time = agv->arrival_at_station[cs_id];
            // 充电开始时间不能早于到达时间和充电站可用时间
            double charging_start = std::max(arrival_time, cs_available_time);
            agv->charging_start_times[j] = charging_start;
            agv->charging_end_times[j] = charging_start + agv->charging_durations[j];
            cs_available_time = agv->charging_end_times[j];
        }

        current_time = cs_available_time;

        for (int i = 0; i < max_waiting_indices.size(); ++i) {
            int first = max_waiting_indices[i].first;
            int second = max_waiting_indices[i].second;
            auto& agv = agv_sol.agvs[first];
            agv->charging_start_times[second] = current_time;
            current_time+=agv->charging_durations[second];
            agv->charging_end_times[second] = current_time;


        }
        //agv_sol.cs_available_time[selectStaion] = earlyCharingtime;



        std::vector<double> agv_current_time(agv_sol.pm->V, 0.0);
        std::vector<double> agv_current_soc(agv_sol.pm->V);
        std::vector<int> agv_current_task(agv_sol.pm->V, 0);
        for (int v = 0; v < agv_sol.pm->V; v++) {
            auto& agv = agv_sol.agvs[v];
            agv_current_soc[v] = agv->initial_soc;

            int num_tasks = agv->task_sequence.size();
            if (num_tasks > 0) {
                agv->task_start_times.resize(num_tasks, 0.0);
                agv->task_end_times.resize(num_tasks, 0.0);
                agv->isCharge.resize(num_tasks, false);
                agv->charging_sessions.resize(num_tasks, {-1, -1});
                agv->charging_start_times.resize(num_tasks, 0.0);
                agv->arrival_at_station.resize(num_tasks, 0.0);
                agv->charging_end_times.resize(num_tasks, 0.0);
                agv->charging_durations.resize(num_tasks, 0.0);

                agv->soc_after_task.resize(num_tasks, 0.0);
                agv->soc_before_task.resize(num_tasks, 0.0);
                agv->soc_at_cs_arrival.resize(num_tasks, 0.0);
                agv->soc_after_charging.resize(num_tasks, 0.0);
                agv->soc_charging_durations.resize(num_tasks, 0.0);
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
            bool flag = false;

            flag = isInMaxWaitingIndices(v,j,max_waiting_indices);
            if (!flag) {
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
                agv_current_soc[v] -= (current_task->cl);

                agv->soc_after_task[j] = agv_current_soc[v];

                // 4. 判断是否需要充电
                if (agv->isCharge[j]) {
                    int station_id = agv->charging_sessions[j].first;
                    double arrival_at_cs = agv->task_end_times[j] + current_task->tr[station_id];
                    double charging_start_time = std::max(arrival_at_cs, agv_sol.cs_available_time[station_id]);
                    double charging_end_time = charging_start_time + agv->charging_durations[j];

                    agv->soc_at_cs_arrival[j] = agv_current_soc[v] - current_task->cr[station_id];
                    if (agv->soc_at_cs_arrival[j]<agv_sol.pm->gamma-1e-3) {
                        cout<<"agv->soc_at_cs_arrival[j]:"<<agv->soc_at_cs_arrival[j]<<endl;
                        cout<<"big error5"<<endl;
                        system("pause");
                    }


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
                    agv_current_soc[v] = agv->soc_after_charging[j]; // 充电后满电
                    agv_current_time[v] = charging_end_time; // 更新车辆当前时间
                } else {
                    agv_current_time[v] = agv->task_end_times[j]; // 更新车辆当前时间
                }

                // 移动到下一个任务
                agv_current_task[v]++;
            }
            else {
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
                agv_current_soc[v] -= (current_task->cl);

                agv->soc_after_task[j] = agv_current_soc[v];

                // 4. 判断是否需要充电
                if (agv->isCharge[j]) {
                    int station_id = agv->charging_sessions[j].first;
                    double arrival_at_cs = agv->task_end_times[j] + current_task->tr[station_id];

                    // 关键修改：确保充电时间不能早于到达充电站的时间
                    double desired_charging_time = agv->charging_start_times[j];  // 你预设的时间
                    double charging_start_time = std::max(arrival_at_cs, desired_charging_time);

                    // 还要考虑充电站的可用时间
                    charging_start_time = std::max(charging_start_time, agv_sol.cs_available_time[station_id]);

                    double charging_end_time = charging_start_time + agv->charging_durations[j];

                    // 其余代码保持不变
                    agv->soc_at_cs_arrival[j] = agv_current_soc[v] - current_task->cr[station_id];
                    if (agv->soc_at_cs_arrival[j] < agv_sol.pm->gamma-1e-3) {
                        cout<<"agv->soc_at_cs_arrival[j]:"<<agv->soc_at_cs_arrival[j]<<endl;
                        cout<<"big error"<<endl;
                        system("pause");
                    }

                    agv->charging_start_times[j] = charging_start_time;
                    agv->charging_end_times[j] = charging_end_time;
                    agv_sol.cs.push_back(std::make_tuple(v, j, charging_start_time,agv->charging_durations[j]));
                    agv_sol.total_charging_sessions++;
                    agv_sol.total_charging_time += agv->charging_durations[j];

                    double waiting_time = charging_start_time - arrival_at_cs;
                    if (waiting_time > 0) {
                        agv_sol.total_waiting_time += waiting_time;
                    }

                    agv_sol.cs_available_time[station_id] = charging_end_time;  // 修复：使用charging_end_time而不是current_time
                    agv_current_soc[v] = agv->soc_after_charging[j];
                    agv_current_time[v] = charging_end_time;
                } else {
                    agv_current_time[v] = agv->task_end_times[j];
                }

                agv_current_task[v]++;
            }


        }

        agv_sol.calculateObjectiveValue();
        //cout<<"agv_sol.getObjectiveValue()4:"<<agv_sol.getObjectiveValue()<<endl;
        // system("pause");
        // 4. 判断目标函数和可行性
        if (agv_sol.getObjectiveValue()< perCost && agv_sol.is_feasible()) {
            return true;
        }
    }
    return false;
}


bool NeighSearch_Random_Ls::isInMaxWaitingIndices(int v, int j, const std::vector<std::pair<int, int>>& max_waiting_indices) {
    return std::find_if(
        max_waiting_indices.begin(),
        max_waiting_indices.end(),
        [v, j](const std::pair<int, int>& p) { return p.first == v && p.second == j; }
    ) != max_waiting_indices.end();
}



