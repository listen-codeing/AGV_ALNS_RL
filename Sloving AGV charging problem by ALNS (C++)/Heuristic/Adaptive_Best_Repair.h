//
// Created by 张欢 on 2025/8/12.
//

#ifndef ADAPTIVE_BEST_REPAIR_H
#define ADAPTIVE_BEST_REPAIR_H



#include "ARepairOperator.h"
#include <vector>
#include <memory>
#include"../random_seed.h"

class AGV_solution;
class AGV;

class Adaptive_Best_Repair : public ARepairOperator {
public:


    // 构造和析构函数
    explicit Adaptive_Best_Repair(std::string s);
    virtual ~Adaptive_Best_Repair();

    // 主要修复函数
    void repairSolution(ISolution& sol) override;

private:

    void processInterval(AGV_solution& agv_sol, int v);


    int selectOptimalChargingStation(AGV_solution& agv_sol, int agv_id, int pos, double current_soc);
    void recalculateCompleteSchedule(AGV_solution& agv_sol);
    void addChargingOperation(AGV_solution& agv_sol,int agv_id, int pos, int station_id);
    double calculateOptimalChargingAmount(AGV_solution& agv_sol,int v,int pos,int station_id);



};



#endif //ADAPTIVE_BEST_REPAIR_H
