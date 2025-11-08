#include "Charging_Worst_Removal.h"


using ChargingDecisionInfo = Charging_Worst_Removal::ChargingDecisionInfo;

Charging_Worst_Removal::Charging_Worst_Removal(double minPerc, double maxPerc, std::string s)
    : ADestroyOperator(minPerc, maxPerc, s) {}

Charging_Worst_Removal::~Charging_Worst_Removal() {}

void Charging_Worst_Removal::destroySolution(ISolution& sol) {

    AGV_solution& agv_sol = dynamic_cast<AGV_solution&>(sol);
    agv_sol.getUncertainty();

    std::vector<ChargingDecisionInfo> charging_decisions;

    for (int v = 0; v < agv_sol.pm->V; v++) {
        auto& agv = agv_sol.agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                ChargingDecisionInfo info       ;
                info.agv_id = v;
                info.task_id = j;
                // 计算task_start_times[j+1] - task_start_times[j]
                if (j + 1 < agv->task_start_times.size()) {
                    info.delta_time = agv->task_start_times[j + 1] - agv->task_start_times[j];
                } else {
                    info.delta_time = 0.0; // 若j+1越界，设为0
                }
                charging_decisions.push_back(info);
            }
        }
    }

    if (charging_decisions.empty()) {
        return;
    }

    size_t total_charging_count = charging_decisions.size();
    size_t min_destroy = static_cast<size_t>(minDestroyPerc * total_charging_count);
    size_t max_destroy = static_cast<size_t>(maxDestroyPerc * total_charging_count);

    min_destroy = std::max(min_destroy, static_cast<size_t>(1));
    max_destroy = std::min(max_destroy, total_charging_count);
    min_destroy = std::min(min_destroy, max_destroy);

    // 按delta_time降序排序
    std::sort(charging_decisions.begin(), charging_decisions.end(),
              [](const ChargingDecisionInfo& a, const ChargingDecisionInfo& b) {
                  return a.delta_time > b.delta_time;
              });

    size_t num_to_remove;
    if (min_destroy == max_destroy) {
        num_to_remove = min_destroy;
    } else {
        num_to_remove = min_destroy + rand() % (max_destroy - min_destroy + 1);
    }

    for (size_t i = 0; i < num_to_remove; i++) {
        removeChargingDecision(agv_sol, charging_decisions[i].agv_id, charging_decisions[i].task_id);
    }
}

void Charging_Worst_Removal::removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id) {
    auto& agv = agv_sol.agvs[agv_id];

    // 记录移除前的充电站信息
    int removed_station = agv->charging_sessions[task_id].first;

    // 清除Truck对象中的充电决策
    agv->isCharge[task_id] = false;
    agv->charging_sessions[task_id] = {-1, -1};
    agv->charging_start_times[task_id] = 0.0;
    agv->charging_end_times[task_id] = 0.0;
    agv->arrival_at_station[task_id] = 0.0;
    agv->soc_at_cs_arrival[task_id] = 0.0;
    agv->charging_durations[task_id] = 0.0;
    // 重新统计剩余的充电决策数量
    agv_sol.total_charging_sessions = 0;
    agv_sol.total_charging_time = 0.0;

    for (int v = 0; v < agv_sol.pm->V; v++) {
        auto& agv = agv_sol.agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                agv_sol.total_charging_sessions++;
                agv_sol.total_charging_time += agv->charging_durations[j];
            }
        }
    }



    // 从充电站使用记录中移除指定的记录
    auto it = std::remove_if(agv_sol.cs.begin(), agv_sol.cs.end(),
        [agv_id, task_id](const std::tuple<int, int, double, int>& record) {
            return std::get<0>(record) == agv_id && std::get<1>(record) == task_id;
        });
    agv_sol.cs.erase(it, agv_sol.cs.end());


}

