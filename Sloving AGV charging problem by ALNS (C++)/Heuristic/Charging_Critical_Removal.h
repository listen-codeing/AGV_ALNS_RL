//
// Created by limin on 2023/10/13.
//
#ifndef AGV_CHARGING_TRUCK_CHARGING_CRITICAL_REMOVAL_H
#define AGV_CHARGING_TRUCK_CHARGING_CRITICAL_REMOVAL_H

#include "ADestroyOperator.h"
#include<random>
#include<algorithm>
#include"../AGV.h"

// 前向声明
class AGV_solution;



class Charging_Critical_Removal: public ADestroyOperator {
public:
    // 充电决策信息结构体（简化版）
    struct ChargingDecisionInfo {
        int agv_id;
        int task_id;
        double waiting_time;    // 唯一的关键性指标：等待时间
    };
    // 构造函数：removeNum作为移除数量参数
    Charging_Critical_Removal(double minNum, double maxNum, std::string s);
    virtual ~Charging_Critical_Removal();

    // 主要的破坏方法
    void destroySolution(ISolution& sol) override;

private:
    // 计算充电决策的等待时间
    double calculateWaitingTime(const AGV_solution& agv_sol, int agv_id, int task_id);

    // 移除充电决策
    void removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id);

};

#endif // TRUCK_CHARGING_CRITICAL_REMOVAL_H