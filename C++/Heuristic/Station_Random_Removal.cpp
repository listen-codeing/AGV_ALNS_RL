#include "Station_Random_Removal.h"
#include "AGV_solution.h"
#include "../random_seed.h"
#include <algorithm>
#include <vector>

// 构造函数，接受最小和最大移除比例
Station_Random_Removal::Station_Random_Removal(double min_perc, double max_perc, std::string s)
    : ADestroyOperator(min_perc, max_perc, s) {
}

Station_Random_Removal::~Station_Random_Removal() {
}

void Station_Random_Removal::destroySolution(ISolution& sol) {
    AGV_solution& agv_sol = dynamic_cast<AGV_solution&>(sol);

    // 1. 找到所有活跃的充电站
    std::vector<int> active_stations;
    std::vector<bool> is_station_active(agv_sol.pm->C, false);
    for (const auto& agv : agv_sol.agvs) {
        for (int j = 0; j < agv->task_sequence.size(); ++j) {
            if (agv->isCharge[j]) {
                int station_id = agv->charging_sessions[j].first;
                if (station_id != -1 && !is_station_active[station_id]) {
                    is_station_active[station_id] = true;
                    active_stations.push_back(station_id);
                }
            }
        }
    }

    if (active_stations.empty()) {
        return; // 没有活跃充电站可供移除
    }

    // 2. 基于活跃充电站数量动态计算删除范围
    size_t total_active_stations = active_stations.size();
    size_t min_destroy = static_cast<size_t>(minDestroyPerc * total_active_stations);
    size_t max_destroy = static_cast<size_t>(maxDestroyPerc * total_active_stations);

    // 确保至少删除1个，最多删除所有活跃充电站
    min_destroy = std::max(static_cast<size_t>(1), min_destroy);
    max_destroy = std::min(total_active_stations, max_destroy);
    min_destroy = std::min(min_destroy, max_destroy);

    // 3. 确定要移除的充电站数量 - 使用随机范围
    size_t num_to_remove;
    if (min_destroy >= max_destroy) {
        num_to_remove = min_destroy;
    } else {
        std::uniform_int_distribution<size_t> dist(min_destroy, max_destroy);
        num_to_remove = dist(sharedGenerator);
    }
    // cout<<"num_to_remove:"<<num_to_remove<<endl;
    // system("pause");
    if (num_to_remove == 0) return;

    // 4. 随机打乱活跃充电站列表
    std::shuffle(active_stations.begin(), active_stations.end(), sharedGenerator);

    // 5. 移除打乱后列表中的前 num_to_remove 个充电站上的所有充电任务
    for (size_t i = 0; i < num_to_remove; ++i) {
        removeAllChargingFromStation(agv_sol, active_stations[i]);
    }
}

void Station_Random_Removal::removeAllChargingFromStation(AGV_solution& agv_sol, int station_id) {
    std::vector<std::pair<int, int>> tasks_to_remove; // (agv_id, task_id)

    for (int v = 0; v < agv_sol.pm->V; ++v) {
        auto& agv = agv_sol.agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); ++j) {
            if (agv->isCharge[j] && agv->charging_sessions[j].first == station_id) {
                tasks_to_remove.push_back({v, j});
            }
        }
    }

    for (const auto& task : tasks_to_remove) {
        removeChargingDecision(agv_sol, task.first, task.second);
    }
}

void Station_Random_Removal::removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id) {
    auto& agv = agv_sol.agvs[agv_id];

    agv->isCharge[task_id] = false;
    agv->charging_sessions[task_id] = {-1, -1};
    agv->charging_start_times[task_id] = 0.0;
    agv->charging_end_times[task_id] = 0.0;
    agv->arrival_at_station[task_id] = 0.0;
    agv->soc_at_cs_arrival[task_id] = 0.0;
    agv->charging_durations[task_id] = 0.0;

    auto it = std::remove_if(agv_sol.cs.begin(), agv_sol.cs.end(),
        [agv_id, task_id](const std::tuple<int, int, double, int>& record) {
            return std::get<0>(record) == agv_id && std::get<1>(record) == task_id;
        });
    agv_sol.cs.erase(it, agv_sol.cs.end());
}
