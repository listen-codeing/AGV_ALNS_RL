//
// Created by 张欢 on 2025/8/14.
//

#ifndef RANDOM_TIME_REPAIR_H
#define RANDOM_TIME_REPAIR_H



#include "ARepairOperator.h"
#include <vector>
#include <memory>

class AGV_solution;
class Truck;

class Random_Time_Repair : public ARepairOperator {
public:

    // 构造和析构函数
    explicit Random_Time_Repair(std::string s);
    virtual ~Random_Time_Repair();

    // 主要修复函数
    void repairSolution(ISolution& sol) override;

private:
    void parallelEventDrivenRepair(AGV_solution& agv_sol);
    int findNextTruck(AGV_solution& agv_sol,
                                               const std::vector<double>& agv_current_time,
                                               const std::vector<int>& agv_current_task);
    void processTaskWithTimeOptimalCharging(AGV_solution& agv_sol, int agv_id, int task_idx,
                                            std::vector<double>& agv_current_time,
                                            std::vector<double>& agv_current_soc,
                                            std::vector<int>& agv_current_task);
    bool mustChargeAfterTask(AGV_solution& agv_sol,int agv_id, int task_idx, double current_soc);
    int selectTimeOptimalStation(AGV_solution& agv_sol,int agv_id, int task_idx,
                                                          double current_soc, double task_end_time);
    double calculateTotalChargingTime(AGV_solution& agv_sol,int agv_id, int task_idx, int station_id,
                                                               double task_end_time, double current_soc);
    bool decideProbabilisticCharging(AGV_solution& agv_sol,int agv_id, double current_soc,
                                                              double current_time, int task_idx);
    void executeChargingOperation(AGV_solution& agv_sol,int agv_id, int task_idx,
                                                           int station_id, double task_end_time,
                                                           std::vector<double>& agv_current_time,
                                                           std::vector<double>& agv_current_soc);
    void calculateTaskTiming(AGV_solution& agv_sol,int agv_id, int task_idx,
                                                      double current_time,
                                                      double& arrival_time,
                                                      double& task_end_time,
                                                      double& energy_consumption);
    void calculateMakespanAndStatistics(AGV_solution& agv_sol,const std::vector<double>& agv_current_time);
    bool getPossibleCharge();

    std::vector<std::tuple<int, int, int>> determin_array;
    bool isMatch(int v, int pos);



};


#endif //RANDOM_TIME_REPAIR_H
