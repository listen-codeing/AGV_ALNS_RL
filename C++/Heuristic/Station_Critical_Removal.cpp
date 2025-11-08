#include "Station_Critical_Removal.h"
#include "AGV_solution.h"
#include "../random_seed.h"
#include <algorithm>
#include <vector>

using StationUsageInfo = Station_Critical_Removal::StationUsageInfo;

// 构造函数，接受最小和最大移除比例
Station_Critical_Removal::Station_Critical_Removal(double min_perc, double max_perc, std::string s)
    : ADestroyOperator(min_perc, max_perc, s) {
}

Station_Critical_Removal::~Station_Critical_Removal() {
}

void Station_Critical_Removal::destroySolution(ISolution& sol) {
    AGV_solution& agv_sol = dynamic_cast<AGV_solution&>(sol);

    // 1. 统计每个充电站的使用情况
    std::vector<StationUsageInfo> station_usage_list;
    std::vector<bool> is_station_collected(agv_sol.pm->C, false);

    for (int v = 0; v < agv_sol.pm->V; v++) {
        auto& agv = agv_sol.agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                int station_id = agv->charging_sessions[j].first;
                if (station_id != -1) {
                    if (!is_station_collected[station_id]) {
                        station_usage_list.push_back({station_id, 0, 0.0});
                        is_station_collected[station_id] = true;
                    }
                    // 找到对应的充电站并更新信息
                    for (auto& usage : station_usage_list) {
                        if (usage.station_id == station_id) {
                            usage.usage_count++;
                            Task* current_task = agv->task_sequence[j];
                            double arrival_at_cs = agv->task_end_times[j] + current_task->tr[station_id];
                            double actual_start_time = agv->charging_start_times[j];
                            double waiting_time = std::max(0.0, actual_start_time - arrival_at_cs);
                            usage.total_waiting_time += waiting_time;
                            break;
                        }
                    }
                }
            }
        }
    }

    if (station_usage_list.empty()) {
        return; // 没有活跃充电站可供移除
    }

    // 2. 基于活跃充电站数量动态计算删除范围
    size_t total_active_stations = station_usage_list.size();
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

    // 4. 按总等待时间降序排序
    std::sort(station_usage_list.begin(), station_usage_list.end(),
              [](const StationUsageInfo& a, const StationUsageInfo& b) {
                  return a.total_waiting_time > b.total_waiting_time;
              });

    // 5. 移除等待时间最长的前 num_to_remove 个充电站上的所有充电任务
    for (size_t i = 0; i < num_to_remove; ++i) {
        removeAllChargingFromStation(agv_sol, station_usage_list[i].station_id);
    }
}

void Station_Critical_Removal::removeAllChargingFromStation(AGV_solution& agv_sol, int station_id) {
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

void Station_Critical_Removal::removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id) {
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
