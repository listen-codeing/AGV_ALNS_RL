//
// Created by limin on 2023/10/13.
//

#ifndef QC_AGV_AGV_RANDOM_REMOVAL_H
#define QC_AGV_AGV_RANDOM_REMOVAL_H


#include "ADestroyOperator.h"
#include<random>
#include<algorithm>
#include"../AGV.h"

// 前向声明
class AGV_solution;

class Charging_Random_Removal: public ADestroyOperator {
public:
    Charging_Random_Removal(double minPerc, double maxPerc, std::string s);
    virtual ~Charging_Random_Removal();

    // 主要的破坏方法
    void destroySolution(ISolution& sol) override;

private:
    void removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id);


};


#endif //QC_AGV_AGV_RANDOM_REMOVAL_H
