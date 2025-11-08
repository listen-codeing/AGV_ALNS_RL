#pragma once

#include "ADestroyOperator.h"
#include "ISolution.h"
#include "AGV_solution.h"
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "../AGV.h"

class AGV_solution;

class Charging_Worst_Removal : public ADestroyOperator {
public:
    struct ChargingDecisionInfo {
        int agv_id;
        int task_id;
        double delta_time; // task_start_times[j+1] - task_start_times[j]
    };

    Charging_Worst_Removal(double minPerc, double maxPerc, std::string s);
    ~Charging_Worst_Removal();

    void destroySolution(ISolution& sol) override;

private:
    void removeChargingDecision(AGV_solution& agv_sol, int agv_id, int task_id);

};