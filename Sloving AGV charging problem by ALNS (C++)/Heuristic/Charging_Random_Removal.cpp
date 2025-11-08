//
// Created by limin on 2023/10/13.
//

#include "Charging_Random_Removal.h"
#include "AGV_solution.h"
#include "assert.h"

Charging_Random_Removal::Charging_Random_Removal(double minPerc, double maxPerc, std::string s) : ADestroyOperator(minPerc, maxPerc, s) {

}

Charging_Random_Removal::~Charging_Random_Removal() {
}

void Charging_Random_Removal::destroySolution(ISolution& sol) {


    AGV_solution& agv_sol = dynamic_cast<AGV_solution&>(sol);
    agv_sol.getUncertainty();
    // 收集所有当前的充电决策
    std::vector<std::pair<int, int>> charging_decisions;
    for (int v = 0; v < agv_sol.pm->V; v++) {
        auto& agv = agv_sol.agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                charging_decisions.push_back({v, j});
            }
        }
    }

    if (charging_decisions.empty()) {
        cout << "No charging decisions to remove" << endl;
        return;
    }

    // 计算移除数量
    size_t total_charging_count = charging_decisions.size();
    size_t min_destroy = static_cast<size_t>(minDestroyPerc * total_charging_count);
    size_t max_destroy = static_cast<size_t>(maxDestroyPerc * total_charging_count);

    min_destroy = std::max(min_destroy, static_cast<size_t>(1));
    max_destroy = std::min(max_destroy, total_charging_count);
    min_destroy = std::min(min_destroy, max_destroy);

    size_t num_to_remove = min_destroy + rand() % (max_destroy - min_destroy + 1);
    // 随机选择并移除充电决策
    shuffle(charging_decisions.begin(), charging_decisions.end(),
            std::default_random_engine(std::random_device{}()));

    std::vector<std::pair<int, int>> removed_decisions;
    for (size_t i = 0; i < num_to_remove; i++) {
        int agv_id = charging_decisions[i].first;
        int task_id = charging_decisions[i].second;
        removeChargingDecision(agv_sol, agv_id, task_id);
        removed_decisions.push_back({agv_id, task_id});
    }



    for (int v = 0; v < agv_sol.pm->V; v++) {
        auto& agv = agv_sol.agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            // if (!agv->isCharge[j]) {
            //     if (agv->charging_sessions[j].first>=0) {
            //         cout<<" error charging_sessions"<<endl;
            //         system("pause");
            //     }
            //
            //
            // }
            if (agv->isCharge[j]) {
                if (agv->charging_sessions[j].first==-1) {
                    cout<<" error2 charging_sessions"<<endl;
                    system("pause");
                }


            }
        }
    }

    // 不计算目标函数，让修复算子处理完整的调度和目标函数计算
}

void Charging_Random_Removal::removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id) {
    auto& agv = agv_sol.agvs[agv_id];

    // 记录移除前的充电站信息
    int removed_station = agv->charging_sessions[task_id].first;

    // 清除Truck对象中的充电决策
    agv->isCharge[task_id] = false;
    //agv->charging_sessions[task_id] = {-1, -1};
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


