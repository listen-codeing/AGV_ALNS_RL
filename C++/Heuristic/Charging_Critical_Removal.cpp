#include "Charging_Critical_Removal.h"
#include "AGV_solution.h"
#include <algorithm>
#include <cstdlib>
#include <ctime>

using ChargingDecisionInfo = Charging_Critical_Removal::ChargingDecisionInfo;

Charging_Critical_Removal::Charging_Critical_Removal(double minPerc, double maxPerc, std::string s) : ADestroyOperator(minPerc, maxPerc, s) {

}

Charging_Critical_Removal::~Charging_Critical_Removal() {
}

void Charging_Critical_Removal::destroySolution(ISolution& sol) {

    AGV_solution& agv_sol = dynamic_cast<AGV_solution&>(sol);
    agv_sol.getUncertainty();
    // 收集所有充电决策及其等待时间信息
    std::vector<ChargingDecisionInfo> charging_decisions;

    for (int v = 0; v < agv_sol.pm->V; v++) {
        auto& agv = agv_sol.agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                ChargingDecisionInfo info;
                info.agv_id = v;
                info.task_id = j;

                // 计算这个充电决策的等待时间
                info.waiting_time = calculateWaitingTime(agv_sol, v, j);

                charging_decisions.push_back(info);
            }
        }
    }

    // 如果没有充电决策，直接返回
    if (charging_decisions.empty()) {
        return;
    }

    // 基于充电数量动态计算删除范围
    size_t total_charging_count = charging_decisions.size();
    size_t min_destroy = static_cast<size_t>(minDestroyPerc * total_charging_count);
    size_t max_destroy = static_cast<size_t>(maxDestroyPerc * total_charging_count);

    // 确保至少删除1个，最多删除所有充电决策
    min_destroy = std::max(min_destroy, static_cast<size_t>(1));
    max_destroy = std::min(max_destroy, total_charging_count);
    min_destroy = std::min(min_destroy, max_destroy);

    // 按等待时间降序排序（等待时间最长的在前面）
    std::sort(charging_decisions.begin(), charging_decisions.end(),
              [](const ChargingDecisionInfo& a, const ChargingDecisionInfo& b) {
                  return a.waiting_time > b.waiting_time;
              });

    // 确定要移除的充电决策数量 - 使用随机范围
    size_t num_to_remove;
    if (min_destroy == max_destroy) {
        num_to_remove = min_destroy;
    } else {
        num_to_remove = min_destroy + rand() % (max_destroy - min_destroy + 1);
    }

    // 移除等待时间最长的充电决策
    for (size_t i = 0; i < num_to_remove; i++) {
        removeChargingDecision(agv_sol, charging_decisions[i].agv_id, charging_decisions[i].task_id);
    }

    //cout << "Removed " << num_to_remove << " charging decisions with highest waiting times" << endl;
}

double Charging_Critical_Removal::calculateWaitingTime(const AGV_solution& agv_sol, int agv_id, int task_id) {
    auto& agv = agv_sol.agvs[agv_id];

    // 获取充电相关信息
    Task* current_task = agv->task_sequence[task_id];
    int charging_station = agv->charging_sessions[task_id].first;

    // 计算到达充电站的时间
    double arrival_at_cs = agv->arrival_at_station[task_id];

    // 获取实际充电开始时间
    double actual_start_time = agv->charging_start_times[task_id];

    // 等待时间 = 实际开始时间 - 到达时间
    double waiting_time = actual_start_time - arrival_at_cs;

    // 确保等待时间非负
    return std::max(0.0, waiting_time);
}

void Charging_Critical_Removal::removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id) {
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
    agv->soc_charging_durations[task_id] = 0.0;
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



    //cout << "Removed charging decision: AGV " << agv_id << " Task " << task_id << " Station " << removed_station << endl;
}

