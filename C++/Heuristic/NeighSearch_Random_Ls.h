//
// Created by 张欢 on 2025/8/1.
//

#ifndef NEIGHSEARCH_RANDOM_LS_H
#define NEIGHSEARCH_RANDOM_LS_H



#include"ILocalSearch.h"
#include"AGV_solution.h"

class NeighSearch_Random_Ls : public ILocalSearch{
public:
    NeighSearch_Random_Ls(std::string s);

    virtual ~NeighSearch_Random_Ls();
    bool performLocalSearch(ISolution &sol);
    std::string getName(){return name;};
    struct ChargingTask {
        int agv_id;
        int task_id;
        int station_id;
        double arrival_time;
        double charging_start_time;
        int original_order;
    };


private:
    std::string name;
    // 核心功能函数
    std::map<int, std::vector<ChargingTask>> analyzeStationQueues(AGV_solution& agv_sol);

    std::vector<std::pair<int, int>> findMostCongestedStations(
        const std::map<int, std::vector<ChargingTask>>& station_queues, int top_k);

    bool performStationReordering(AGV_solution& agv_sol, int station_id,
                                  std::vector<ChargingTask>& station_queue);

    void restoreOriginalOrdering(AGV_solution& agv_sol, int station_id,
                                 const std::vector<ChargingTask>& original_queue);

    void recalculateScheduleAfterReordering(AGV_solution& agv_sol);
    void recalculateFromTimePoint(AGV_solution& agv_sol, double start_time);
    bool isInMaxWaitingIndices(int v, int j, const std::vector<std::pair<int, int>>& max_waiting_indices);

};


#endif //NEIGHSEARCH_RANDOM_LS_H
