//
// Created by limin on 2023/10/13.
//
#ifndef TRUCK_STATION_CTITICAL_REMOVAL_H
#define TRUCK_STATION_CTITICAL_REMOVAL_H

#include "ADestroyOperator.h"
#include"../AGV.h"
#include <vector>
#include<algorithm>

// 前向声明
class AGV_solution;

class Station_Critical_Removal: public ADestroyOperator {
public:
    // 充电站使用信息结构体
    struct StationUsageInfo {
        int station_id;
        int usage_count;
        double total_waiting_time;
    };
    
    // 构造函数：按比例范围移除充电站
    Station_Critical_Removal(double min_perc, double max_perc, std::string s);
    virtual ~Station_Critical_Removal();
    
    // 主要的破坏方法
    void destroySolution(ISolution& sol) override;

private:
    // 根据统计数据选择要移除的多个充电站
    std::vector<int> selectStationsToRemove(const std::vector<StationUsageInfo>& station_usage, size_t num_to_remove);

    // 移除指定充电站的所有充电决策
    void removeAllChargingFromStation(AGV_solution& agv_sol, int station_id);

    // 移除单个充电决策
    void removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id);
};

#endif // TRUCK_STATION_CTITICAL_REMOVAL_H
