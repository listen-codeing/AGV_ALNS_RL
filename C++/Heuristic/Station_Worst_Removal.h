//
// Created by limin on 2023/10/13.
//

#ifndef TRUCK_STATION_WORST_REMOVAL_H
#define TRUCK_STATION_WORST_REMOVAL_H

#include "ADestroyOperator.h"
#include <vector>
#include <string>
#include"../AGV.h"

#include<algorithm>

class AGV_solution; // 前向声明

class Station_Worst_Removal : public ADestroyOperator {
public:
    // 充电站使用信息结构体
    struct StationUsageInfo {
        int station_id;
        double total_waiting_time;
    };

    // 构造函数：按比例范围移除最差的充电站
    Station_Worst_Removal(double min_perc, double max_perc, std::string s);
    virtual ~Station_Worst_Removal();

    // 主要的破坏方法
    void destroySolution(ISolution& sol) override;

private:
    // 移除指定充电站的所有充电决策
    void removeAllChargingFromStation(AGV_solution& agv_sol, int station_id);

    // 移除单个充电决策
    void removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id);
};

#endif //TRUCK_STATION_WORST_REMOVAL_H
