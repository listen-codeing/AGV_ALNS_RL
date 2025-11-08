//
// Created by limin on 13/4/2025.
//

#include "AGV_solution.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>  // ??????????????????
#include <vector>   // ????????????
#include <numeric>
#include <algorithm>
#include <set>
#include "../AGV.h"
#include "../TestUnit.h"
#include "../random_seed.h"

AGV_solution::AGV_solution(Parameter *agv_pm, string model_name) {
    this->pm = agv_pm;
    this->solution_name = model_name;
    this->startTime = clock();
    this->UB = DBL_MAX;
    this->LB = 0;
    initialize();
}

void AGV_solution::initialize() {
    // ?????????AGV
    agvs.clear();
    for (int v = 0; v < pm->V; ++v) {
        agvs.push_back(std::make_shared<AGV>(*pm, v));
    }
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];

        int num_tasks = agv->task_sequence.size();


        agv->task_start_times.resize(num_tasks, 0.0);
        agv->task_end_times.resize(num_tasks, 0.0);
        agv->isCharge.resize(num_tasks, false);
        agv->charging_sessions.resize(num_tasks, {-1, -1});
        agv->charging_start_times.resize(num_tasks, 0.0);
        agv->arrival_at_station.resize(num_tasks, 0.0);
        agv->charging_end_times.resize(num_tasks, 0.0);
        agv->charging_durations.resize(num_tasks, 0.0);

        agv->soc_after_task.resize(num_tasks, 0.0);
        agv->soc_before_task.resize(num_tasks, 0.0);
        agv->soc_at_cs_arrival.resize(num_tasks, 0.0);
        agv->soc_after_charging.resize(num_tasks, 0.0);
        agv->soc_charging_durations.resize(num_tasks, 0.0);


    }
    pm->cs_available_time = std::vector<double>(pm->C, 0.0);

    // ?????Solution???е???????
    total_charging_time = 0.0;
    total_travel_time = 0.0;
    total_waiting_time = 0.0;
    total_charging_sessions = 0;
    makespan = 0.0;
    cs_available_time = std::vector<double>(pm->C, 0.0);
    cs.clear();
}
void AGV_solution::getInitial1() {

    initialize();

    std::vector<double> agv_current_time(pm->V, 0.0);
    std::vector<double> agv_current_soc(pm->V);
    std::vector<int> agv_current_task(pm->V, 0);
    // ????????????????????
    for (int v = 0; v < pm->V; v++) {
        agv_current_soc[v] = agvs[v]->initial_soc;
    }
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        agv_current_soc[v] = agv->initial_soc;


        int num_tasks = pm->J_v[v];
        if (num_tasks > 0) {
            agv->task_arrival_times.resize(num_tasks, 0.0);
            agv->task_start_times.resize(num_tasks, 0.0);
            agv->task_end_times.resize(num_tasks, 0.0);
            agv->isCharge.resize(num_tasks, false);
            agv->soc_after_task.resize(num_tasks, 0.0);
            agv->soc_before_task.resize(num_tasks, 0.0);
            agv->charging_start_times.resize(num_tasks, 0.0);
            agv->charging_end_times.resize(num_tasks, 0.0);
            agv->soc_at_cs_arrival.resize(num_tasks, 0.0);
            agv->arrival_at_station.resize(num_tasks, 0.0);
            agv->charging_durations.resize(num_tasks, 0.0);
            agv->soc_charging_durations.resize(num_tasks, 0.0);
            agv->charging_sessions.resize(num_tasks, {-1, -1});
        }
    }

    // ???????????е???
    while (true) {
        // ??????????????????????????????????????

        int next_agv = findNextTruck(agv_current_time, agv_current_task);

        if (next_agv == -1) break; // ???????????

        // ??????п???????????
        processTaskWithTimeOptimalCharging(next_agv, agv_current_task[next_agv],
                                         agv_current_time, agv_current_soc, agv_current_task);


    }
    calculateObjectiveValue();

}

int AGV_solution::findNextTruck(const std::vector<double>& agv_current_time,const std::vector<int>& agv_current_task) {
    int next_agv = -1;
    double earliest_time = std::numeric_limits<double>::max();

    // ???????п???????????????????????δ???????????
    for (int v = 0; v < pm->V; v++) {
        // ????????????δ????????
        if (agv_current_task[v] < agvs[v]->task_sequence.size()) {
            // ???????????????????磬???????
            if (agv_current_time[v] < earliest_time) {
                earliest_time = agv_current_time[v];
                next_agv = v;
            }
        }
    }

    return next_agv; // ??????п?????????????????-1
}

void AGV_solution::processTaskWithTimeOptimalCharging( int agv_id, int task_idx,std::vector<double>& agv_current_time,std::vector<double>& agv_current_soc,
    std::vector<int>& agv_current_task) {

    auto& agv = agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    // 1. ?????????е??????
    double arrival_time, task_start_time, task_end_time , energy_consumption;

    calculateTaskTiming(agv_id, task_idx, agv_current_time[agv_id],
                        arrival_time, task_start_time, task_end_time, energy_consumption);

    // 2. ?????????
    agv_current_soc[agv_id] -= energy_consumption;
    agv->soc_before_task[task_idx] = agv_current_soc[agv_id];

    agv_current_soc[agv_id] -= (current_task->cl);
    agv->soc_after_task[task_idx] = agv_current_soc[agv_id];

    agv->task_arrival_times[task_idx] = arrival_time;
    agv->task_start_times[task_idx] = task_start_time;
    agv->task_end_times[task_idx] = task_end_time;
    agv_current_time[agv_id] = task_end_time;


    // 3. ?????????
    bool need_charging = false;
    int selected_station = -1;
    bool must_charge = mustChargeAfterTask(agv_id, task_idx, agv->soc_after_task[task_idx]);

    if (must_charge) {
        need_charging = true;
        // ???????????????к?????
        selected_station = selectTimeOptimalStation(agv_id, task_idx,agv->soc_after_task[task_idx], task_end_time);
    }

    // 4. ??г?????
    if (need_charging && selected_station != -1) {
        executeChargingOperation(agv_id, task_idx, selected_station,
                               task_end_time, agv_current_time, agv_current_soc);
        agv_current_soc[agv_id] = agv->soc_after_charging[task_idx];
    } else {
        agv_current_time[agv_id] = task_end_time;
        agv_current_soc[agv_id] = agv->soc_after_task[task_idx];
    }

    // 5. ??????????????
    agv_current_task[agv_id]++;
}

int AGV_solution::selectTimeOptimalStation(int agv_id, int task_idx,double current_soc, double task_end_time) {

    auto& agv = agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    int best_station = -1;
    double min_total_time = std::numeric_limits<double>::max();

    std::vector<int> feasible_stations;

    for (int c = 0; c < pm->C; c++) {
        // ??????????????????
        double energy_to_station = current_task->cr[c];
        double soc_at_station = current_soc - energy_to_station;

        if (soc_at_station < pm->gamma-1e-3) {

            continue; // ?????????????????
        }

        feasible_stations.push_back(c);

        // ?????????
        double total_time = calculateTotalChargingTime(agv_id, task_idx, c,
                                                      task_end_time, current_soc);

        if (total_time < min_total_time) {
            min_total_time = total_time;
            best_station = c;
        }
    }

    // ????????????е???????????????????
    if (best_station == -1)
    {
        cout<<"error4"<<endl;
        system("pause");
        // ???????п??????????-1?????????
        return 0;
    }


    return best_station;
}

double AGV_solution::calculateTotalChargingTime(int agv_id, int task_idx, int station_id, double task_end_time,
                                                double current_soc) {

    auto& agv = agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    // 1. ????????????
    double travel_to_station_time = current_task->tr[station_id];

    // 2. ????????????
    double arrival_at_station = task_end_time + travel_to_station_time;

    // 3. ????????????
    double charging_start_time = std::max(arrival_at_station, cs_available_time[station_id]);
    double waiting_time = charging_start_time - arrival_at_station;
    //?????????????????
    // 4. ??????
    double charging_time = pm->tu;

    // 5. ?????????????????????????????????????????
    double travel_to_next_time = 0.0;
    if (task_idx < agv->task_sequence.size() - 1) {
        Task* next_task = agv->task_sequence[task_idx + 1];
        travel_to_next_time = next_task->tf[station_id];
    }

    // ????? = ?????????? + ?????? + ?????? + ????????????
    return travel_to_station_time + waiting_time + charging_time + travel_to_next_time;
}


bool AGV_solution::mustChargeAfterTask(int agv_id, int task_idx, double current_soc) {
    auto& agv = agvs[agv_id];
    auto& current_task = agv->task_sequence[task_idx];
    if (task_idx >= agv->task_sequence.size() - 1) {
        return false; // ?????????????????
    }
    if (agv->isCharge[task_idx]) {
        return false;
    }
    bool need_charge = false;
    if (task_idx<agv->soc_before_task.size()-2) {
        // ???????????????????/???????????
        double next_soc = simulateTaskExecution(agv_id, task_idx+1, current_soc);

        next_soc -= (agv->task_sequence[task_idx + 1]->cl);
        if (agv->isCharge[task_idx+1]) {
            double min_soc_after_next = next_soc;
            double min_soc_to_station = agv->task_sequence[task_idx+1]->cr[agv->charging_sessions[task_idx + 1].first];
            min_soc_after_next = min_soc_after_next - min_soc_to_station;
            if (min_soc_after_next < pm->gamma-1e-3) {
                need_charge = true;
            }
        }
        else {
            // ??У???????+???????????/??????????????????? gamma
            double min_soc_after_next = next_soc;
            double min_soc_to_station = std::numeric_limits<double>::max();
            for (int c = 0; c < pm->C; c++) {
                double soc_to_station = agv->task_sequence[task_idx + 1]->cr[c];
                if (soc_to_station < min_soc_to_station) {
                    min_soc_to_station = soc_to_station;
                }
            }
            min_soc_after_next = min_soc_after_next - min_soc_to_station;
            if (min_soc_after_next < pm->gamma-1e-3) {
                need_charge = true;

            }
        }
    }
    else {
        // ???????????????????/???????????
        double next_soc = simulateTaskExecution(agv_id, task_idx+1, current_soc);
        next_soc -= (agv->task_sequence[task_idx + 1]->cl);
        // ??У???????+???????????/??????????????????? gamma
        double min_soc_after_next = next_soc;
        if (min_soc_after_next < pm->gamma-1e-3) {
            need_charge = true;
        }
    }

    return need_charge;
}

void AGV_solution::calculateTaskTiming(int agv_id, int task_idx, double current_time, double &arrival_time,
                                       double &task_start_time, double &task_end_time, double &energy_consumption) {
    auto& agv = agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    // ??????????????????????
    double travel_time_to_task = 0.0;

    if (task_idx == 0) {
        // ?????????????λ?ó???
        travel_time_to_task = current_task->te;
        energy_consumption = current_task->ce;
    } else {
        // ???????????????????????????????
        if (agv->isCharge[task_idx - 1]) {
            // ?????????г?磬?????????
            int charging_station = agv->charging_sessions[task_idx - 1].first;
            travel_time_to_task = current_task->tf[charging_station];
            energy_consumption = current_task->cf[charging_station];
        } else {
            // ??????????г?磬??????????????λ?ó???
            travel_time_to_task = current_task->te;
            energy_consumption = current_task->ce;
        }
    }

    // ???????
    arrival_time = current_time + travel_time_to_task;
    task_start_time = max(current_task->ready_time, arrival_time);
    task_end_time = task_start_time + current_task->tl;
}

void AGV_solution::executeChargingOperation(int agv_id, int task_idx,int station_id, double task_end_time,std::vector<double>& agv_current_time,
                                                           std::vector<double>& agv_current_soc) {
    auto& agv = agvs[agv_id];
    Task* current_task = agv->task_sequence[task_idx];

    // 1. ???ó?????????
    agv->isCharge[task_idx] = true;
    agv->charging_sessions[task_idx] = {station_id, -1};


    // 3. ???????????
    double arrival_at_cs = task_end_time + current_task->tr[station_id];
    double charging_start_time = std::max(arrival_at_cs, cs_available_time[station_id]);
    agv->soc_at_cs_arrival[task_idx] = agv->soc_after_task[task_idx]-agv->task_sequence[task_idx]->cr[station_id];
    agv->isCharge[task_idx] = true;
    agv->charging_sessions[task_idx] = {station_id, -1};


    double min_charge = 0;
    double need_consumption = 0.0;
    double min_to_station = std::numeric_limits<double>::max();
    if (task_idx < agv->soc_after_charging.size() - 2) {
        need_consumption+= agv->task_sequence[task_idx + 1]->cf[station_id];
        need_consumption+= agv->task_sequence[task_idx + 1]->cl;
        for (int c=0;c<pm->C;c++) {
            if (min_to_station>agv->task_sequence[task_idx + 1]->cr[station_id]) {
                min_to_station=agv->task_sequence[task_idx + 1]->cr[station_id];
            }
        }
        need_consumption+= min_to_station;
        min_charge = need_consumption+ pm->gamma - agv->soc_after_task[task_idx];

    }
    else {
        min_charge = agv->task_sequence[task_idx + 1]->cf[station_id]+agv->task_sequence[task_idx + 1]->cl;
    }

    // 1. ????????磬?????????????????????????????????gamma
    double max_charge = pm->Pi - agv->soc_at_cs_arrival[task_idx];
    double cur_soc = agv->soc_at_cs_arrival[task_idx] + max_charge;
    int prev_station = station_id;
    bool feasible = true;
    for (int i = task_idx + 1; i < agv->task_sequence.size(); ++i) {
        Task* t = agv->task_sequence[i];
        if (i == task_idx + 1) {
            cur_soc -= t->cf[prev_station];
        } else if (agv->isCharge[i - 1]) {
            int prev_cs = agv->charging_sessions[i - 1].first;
            cur_soc -= t->cf[prev_cs];
        } else {
            cur_soc -= t->ce;
        }
        cur_soc -= t->cl;
        // ??к??????????????????cr
        if (cur_soc < pm->gamma-1e-3) {
            feasible = false;
            break;
        }
    }
    // 2. ?ж??????????
    if (feasible && cur_soc >= pm->gamma-1e-3) {
        // ????????????????????
        double total_consume = 0.0;
        prev_station = station_id;
        for (int i = task_idx + 1; i < agv->task_sequence.size(); ++i) {
            Task* t = agv->task_sequence[i];
            if (i == task_idx + 1) {
                total_consume += t->cf[prev_station];
            } else if (agv->isCharge[i - 1]) {
                int prev_cs = agv->charging_sessions[i - 1].first;
                total_consume += t->cf[prev_cs];
            } else {
                total_consume += t->ce;
            }
            total_consume += t->cl;
        }
        double required_charge = pm->gamma + total_consume - agv->soc_at_cs_arrival[task_idx];
        //if (required_charge < 0) required_charge = 0;

        if (required_charge > max_charge) required_charge = max_charge;

        agv->soc_charging_durations[task_idx] = required_charge;
        agv->soc_after_charging[task_idx] = agv->soc_at_cs_arrival[task_idx] + required_charge;
        //  ?????????
//        if (pm->tm->warm_start) {
//            agv->charging_durations[task_idx] = pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx], agv->soc_after_charging[task_idx]) + pm->tu;
//
//        }
//        else {
//            agv->charging_durations[task_idx] = agv->soc_charging_durations[task_idx] /pm->eta+ pm->tu;
//
//        }
        agv->charging_durations[task_idx] = pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx], agv->soc_after_charging[task_idx]) + pm->tu;
        //agv->charging_durations[task_idx] = agv->soc_charging_durations[task_idx] /pm->eta+ pm->tu;

    } else {
        // ??????
        //agv->soc_charging_durations[task_idx] = max_charge;
        double optimal_charging = pm->get_best_soc_delta(agv->soc_at_cs_arrival[task_idx]);
        //double optimal_charging = (pm->Pi-agv->soc_at_cs_arrival[task_idx])*0.5;
        //cout<<"optimal_charging:"<<optimal_charging<<endl;

        agv->soc_charging_durations[task_idx] = max(min_charge,optimal_charging+agv->soc_at_cs_arrival[task_idx]);

        agv->soc_after_charging[task_idx] = agv->soc_at_cs_arrival[task_idx] + agv->soc_charging_durations[task_idx];
//        if (pm->tm->warm_start) {
//            agv->charging_durations[task_idx] = pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx], agv->soc_after_charging[task_idx]) + pm->tu;
//
//        }
//        else {
//            agv->charging_durations[task_idx]  = agv->soc_charging_durations[task_idx] /pm->eta+ pm->tu;
//
//        }
        agv->charging_durations[task_idx] = pm->get_charge_time_min(agv->soc_at_cs_arrival[task_idx], agv->soc_after_charging[task_idx]) + pm->tu;
        //???????????
        //agv->charging_durations[task_idx]  = agv->soc_charging_durations[task_idx] /pm->eta+ pm->tu;
    }


    double charging_end_time = charging_start_time + agv->charging_durations[task_idx];

    // 4. ????????????
    agv->arrival_at_station[task_idx] = arrival_at_cs;
    agv->charging_start_times[task_idx] = charging_start_time;
    agv->charging_end_times[task_idx] = charging_end_time;

    // ??????????????к???????agv->soc_after_task[task_idx]????????????????
    agv->soc_at_cs_arrival[task_idx] = agv->soc_after_task[task_idx] - current_task->cr[station_id];



    // 5. ??????????

    total_charging_sessions++;
    total_charging_time += agv->charging_durations[task_idx];

    double waiting_time = charging_start_time - arrival_at_cs;
    if (waiting_time > 0) {
        total_waiting_time += waiting_time;
    }

    // 6. ??????????????
    cs_available_time[station_id] = charging_end_time;

    // 7. ?????????
    agv_current_soc[agv_id] = agv->soc_after_charging[task_idx]; // ????????
    agv_current_time[agv_id] = charging_end_time;   // ?????????????

}




void AGV_solution::getInitial() {

    // ??????????????
    initialize();

    // ?????Σ???????????????????
    for (int v = 0; v < pm->V; ++v) {
        auto& agv = agvs[v];

        double soc = agv->initial_soc;

        for (int j = 0; j < agv->task_sequence.size() - 1; ++j) {

            Task* task = agv->task_sequence[j];

            // ????????????????
            if (j == 0) {
                soc -= task->ce;
            } else if (agv->isCharge[j - 1]) {

                int cs = agv->charging_sessions[j - 1].first;

                soc -= task->cf[cs];

            } else {

                soc -= task->ce;
            }

            // ???????????
            soc -= task->cl;


            // ?ж??????????
            bool need_charge = false;
            int selected_cs = -1;
            if (j<agv->soc_before_task.size()-2) {

                // ???????????????????/???????????
                double next_soc = simulateTaskExecution(v, j+1, soc);

                next_soc -= (agv->task_sequence[j + 1]->cl);
                if (agv->isCharge[j+1]) {
                    double min_soc_after_next = next_soc;
                    double min_soc_to_station = agv->task_sequence[j+1]->cr[agv->charging_sessions[j + 1].first];
                    min_soc_after_next = min_soc_after_next - min_soc_to_station;
                    if (min_soc_after_next < pm->gamma-1e-3) {
                        need_charge = true;
                    }
                }
                else {
                    // ??У???????+???????????/??????????????????? gamma
                    double min_soc_after_next = next_soc;
                    double min_soc_to_station = std::numeric_limits<double>::max();
                    for (int c = 0; c < pm->C; c++) {
                        double soc_to_station = agv->task_sequence[j + 1]->cr[c];
                        if (soc_to_station < min_soc_to_station) {
                            min_soc_to_station = soc_to_station;
                        }
                    }
                    min_soc_after_next = min_soc_after_next - min_soc_to_station;

                    if (min_soc_after_next < pm->gamma-1e-3) {

                        need_charge = true;

                    }
                }
            }
            else {
                // ???????????????????/???????????
                double next_soc = simulateTaskExecution(v, j+1, soc);
                next_soc -= (agv->task_sequence[j + 1]->cl);
                // ??У???????+???????????/??????????????????? gamma
                double min_soc_after_next = next_soc;
                if (min_soc_after_next < pm->gamma-1e-3) {

                    need_charge = true;
                }
            }

            if (need_charge) {
                selected_cs = selectOptimalChargingStation(v, j, soc);
                agv->isCharge[j] = true;
                agv->charging_sessions[j] = {selected_cs, -1};

                agv->soc_after_charging[j] = pm->Pi; // ????????
                soc = pm->Pi;
            }
        }
    }

    cs_available_time.assign(pm->C, 0.0);
    total_waiting_time = 0.0;
    total_charging_time = 0.0;
    total_charging_sessions = 0;
    cs.clear();


    // ????????????????
    std::vector<double> agv_current_time(pm->V, 0.0);
    std::vector<double> agv_current_soc(pm->V);
    std::vector<int> agv_current_task(pm->V, 0);

    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        agv_current_soc[v] = agv->initial_soc;

        int num_tasks = pm->J_v[v];
        if (num_tasks > 0) {
            agv->task_start_times.resize(num_tasks, 0.0);
            agv->task_end_times.resize(num_tasks, 0.0);
            agv->isCharge.resize(num_tasks, false);
            agv->soc_after_task.resize(num_tasks, 0.0);
            agv->soc_before_task.resize(num_tasks, 0.0);
            agv->charging_start_times.resize(num_tasks, 0.0);
            agv->charging_end_times.resize(num_tasks, 0.0);
            agv->soc_at_cs_arrival.resize(num_tasks, 0.0);
            agv->arrival_at_station.resize(num_tasks, 0.0);
            agv->charging_durations.resize(num_tasks, 0.0);
            agv->soc_charging_durations.resize(num_tasks, 0.0);
            agv->charging_sessions.resize(num_tasks, {-1, -1});
        }
    }

    // ???????????е???
    while (true) {
        // ????????????????????????δ???????
        int next_agv = -1;
        double earliest_time = std::numeric_limits<double>::max();

        for (int v = 0; v < pm->V; v++) {
            if (agv_current_task[v] < agvs[v]->task_sequence.size()) {
                if (agv_current_time[v] < earliest_time) {
                    earliest_time = agv_current_time[v];
                    next_agv = v;
                }
            }
        }

        // ?????д????????????????
        if (next_agv == -1) break;

        // ??????г???????????
        int v = next_agv;
        int j = agv_current_task[v];
        auto& agv = agvs[v];
        Task* current_task = agv->task_sequence[j];

        // 1. ??????????????????????
        double travel_time_to_task = 0.0;
        double energy_consumption_to_task = 0.0;

        if (j == 0) {
            travel_time_to_task = current_task->te;
            energy_consumption_to_task = current_task->ce;
        } else {
            if (agv->isCharge[j-1]) {
                int charging_station = agv->charging_sessions[j-1].first;
                travel_time_to_task = current_task->tf[charging_station];
                energy_consumption_to_task = current_task->cf[charging_station];
            } else {
                travel_time_to_task = current_task->te;
                energy_consumption_to_task = current_task->ce;
            }
        }

        // 2. ?????????????????
        agv->task_start_times[j] = agv_current_time[v] + travel_time_to_task;
        agv->task_end_times[j] = agv->task_start_times[j] + current_task->tl;

        // 3. ?????????
        agv_current_soc[v] -= energy_consumption_to_task;

        agv->soc_before_task[j] = agv_current_soc[v];
        agv_current_soc[v] -= (current_task->cl);

        agv->soc_after_task[j] = agv_current_soc[v];

        // 4. ?ж??????????
        if (agv->isCharge[j]) {
            if (!isLastCharging(v,j)) {
                int station_id = agv->charging_sessions[j].first;
                double arrival_at_cs = agv->task_end_times[j] + current_task->tr[station_id];
                double charging_start_time = std::max(arrival_at_cs, cs_available_time[station_id]);


                agv->arrival_at_station[j] = arrival_at_cs;
                agv->soc_at_cs_arrival[j] = agv_current_soc[v] - current_task->cr[station_id];

                if (agv->soc_at_cs_arrival[j]<pm->gamma-1e-3) {

                    cout<<"error2"<<endl;
                    system("pause");
                }
                agv->soc_after_charging[j] = pm->Pi; // ????????
                double need_energy = agv->soc_after_charging[j] - agv->soc_at_cs_arrival[j];;

                // ?????????????λ??С???
                double need_charging_time = pm->get_charge_time_min(agv->soc_at_cs_arrival[j], agv->soc_after_charging[j]);
//                double need_charging_time;
//                if (pm->tm->warm_start) {
//                    need_charging_time = pm->get_charge_time_min(agv->soc_at_cs_arrival[j], agv->soc_after_charging[j]);
//                }
//                else {
//                    need_charging_time = (agv->soc_after_charging[j]-agv->soc_at_cs_arrival[j])/pm->eta;
//                }
                need_charging_time += pm->tu; // ?????????

                double charging_end_time = charging_start_time + need_charging_time;
                agv->charging_start_times[j] = charging_start_time;
                agv->charging_end_times[j] = charging_end_time;

                agv->soc_charging_durations[j] = need_energy;
                agv->charging_durations[j] = need_charging_time;
                cs.push_back(std::make_tuple(v, j, charging_start_time, need_charging_time));
                total_charging_sessions++;
                total_charging_time += need_charging_time;

                double waiting_time = charging_start_time - arrival_at_cs;
                if (waiting_time > 0) {
                    total_waiting_time += waiting_time;
                }

                cs_available_time[station_id] = charging_end_time;
                agv_current_soc[v] = pm->Pi; // ????????
                agv_current_time[v] = charging_end_time; // ?????????????
            }
            else {
                int station_id = agv->charging_sessions[j].first;
                double arrival_at_cs = agv->task_end_times[j] + current_task->tr[station_id];
                double charging_start_time = std::max(arrival_at_cs, cs_available_time[station_id]);
                agv->arrival_at_station[j] = arrival_at_cs;
                agv->soc_at_cs_arrival[j] = agv_current_soc[v] - current_task->cr[station_id];
                if (agv->soc_at_cs_arrival[j] < pm->gamma-1e-3) {
                    cout<<"error3"<<endl;
                    system("pause");
                }

                // ???????????????????????????????
                double energy_consumption = 0;
                for (int k = j + 1; k < agv->task_sequence.size(); ++k) {
                    Task* t = agv->task_sequence[k];
                    if (k == j + 1) {
                        energy_consumption += t->cf[station_id]; // ????????????????????
                    } else {
                        energy_consumption += t->ce; // ?????????
                    }
                    energy_consumption += t->cl; // ??????????
                }
                double need_energy = pm->gamma +  energy_consumption - agv->soc_at_cs_arrival[j]; // ??????????gamma

                // if (need_energy < 0) {
                //     cout<<"error4"<<endl;
                //     //system("pause");
                //
                // }
                if (need_energy > pm->Pi - agv->soc_at_cs_arrival[j]) {
                    cout<<"error5"<<endl;
                    //system("pause");
                }
                int soc_start = static_cast<int>(std::round(agv->soc_at_cs_arrival[j]));
                int soc_end = static_cast<int>(std::round(agv->soc_at_cs_arrival[j] + need_energy));
                double need_charging_time = pm->get_charge_time_min(soc_start, soc_end);
//                double need_charging_time;
//                if (pm->tm->warm_start) {
//                    need_charging_time = pm->get_charge_time_min(soc_start, soc_end);
//                }
//                else {
//                    need_charging_time = (soc_end-soc_start)/pm->eta;
//                }


                need_charging_time += pm->tu;
                double charging_end_time = charging_start_time + need_charging_time;
                agv->charging_start_times[j] = charging_start_time;
                agv->charging_end_times[j] = charging_end_time;
                agv->soc_after_charging[j] = agv->soc_at_cs_arrival[j] + need_energy;
                agv->soc_charging_durations[j] = need_energy;
                agv->charging_durations[j] = need_charging_time;
                cs.push_back(std::make_tuple(v, j, charging_start_time, need_charging_time));
                total_charging_sessions++;
                total_charging_time += need_charging_time;

                double waiting_time = charging_start_time - arrival_at_cs;
                if (waiting_time > 0) {
                    total_waiting_time += waiting_time;
                }

                cs_available_time[station_id] = charging_end_time;
                agv_current_soc[v] = agv->soc_after_charging[j];
                agv_current_time[v] = charging_end_time;
            }

        } else {
            agv_current_soc[v] = agv->soc_after_task[j];
            agv_current_time[v] = agv->task_end_times[j]; // ?????????????
        }

        // ??????????????
        agv_current_task[v]++;
    }

    // ????makespan
    makespan = 0.0;
    //
    for (int v = 0; v < pm->V; v++) {
        if (!agvs[v]->task_sequence.empty()) {
            makespan = std::max(makespan, agv_current_time[v]);
        }
    }
    calculateObjectiveValue();
}
bool AGV_solution::isLastCharging(int v, int j) {
    auto& agv = agvs[v];
    for (int k = j + 1; k < agv->isCharge.size(); ++k) {
        if (agv->isCharge[k]) {
            return false;
        }
    }
    return agv->isCharge[j];
}

void AGV_solution::getUncertainty() {
    if (this->pm->tm->model_type == ModelType::DRO) {
        // ???DRO????????????????????????????????
        xi_uncertainty = pm->xi_ave_dro;      // ???????????????tl??
        zeta_uncertainty = pm->zeta_cvar_dro;  // ????????????????cl??
    } else if (this->pm->tm->model_type == ModelType::SAA) {
        // ???SAA????????????????????????????????
        xi_uncertainty = pm->xi_ave_saa;      // ???????????????tl??
        zeta_uncertainty = pm->zeta_ave_saa;  // ????????????????cl??
    } else {
        // ???????DRO??SAA????????ò??????
        xi_uncertainty = 0.0;
        zeta_uncertainty = 0.0;
    }
}

double AGV_solution::calculateMinEnergyForNextTask(int agv_id, int current_task_idx) {
    if (current_task_idx >= pm->J_v[agv_id] - 1) {
        return 0.0;
    } else if (current_task_idx == pm->J_v[agv_id] - 2) {
        Task* next_task = agvs[agv_id]->task_sequence[current_task_idx + 1];
        double energy_to_next = next_task->ce;
        double energy_for_next = next_task->cl;
        return energy_to_next + energy_for_next;
    } else if (current_task_idx < pm->J_v[agv_id] - 2) {
        Task* next_task = agvs[agv_id]->task_sequence[current_task_idx + 1];
        double energy_to_next = next_task->ce;
        double energy_for_next = next_task->cl;
        double min_energy_to_cs = findMinEnergyToChargingStation(agv_id, current_task_idx + 1).first;
        return energy_to_next + energy_for_next + min_energy_to_cs;
    }
    // ???????????
    return 0.0;
}

//?????????
int AGV_solution::selectOptimalChargingStation(int agv_id, int pos, double current_soc) {


    Task* current_task = agvs[agv_id]->task_sequence[pos];

    vector<int> feasible_stations;

    // ????????????
    Task* next_task = agvs[agv_id]->task_sequence[pos+1];


    // ??????????????????
    for (int c = 0; c < pm->C; c++) {
        // ???????????????????????

        double energy_to_station = current_task->cr[c];

        double soc_after_charging = current_soc - energy_to_station;


        // ??????????????
        if (soc_after_charging  >= pm->gamma-1e-3) {
            feasible_stations.push_back(c);

        }
    }


    // ?????п??е??????????????
    if (feasible_stations.empty()) {

        cout<<"error1"<<endl;
        system("pause");
    }

    // ????????????е????
    // static std::mt19937 rng(42); // ??ù?????? 42
    // std::uniform_int_distribution<int> dist(0, feasible_stations.size() - 1);
    // int selected_station = feasible_stations[dist(rng)];

    std::uniform_int_distribution<int> dist(0, feasible_stations.size() - 1);
    int selected_station = feasible_stations[dist(sharedGenerator)]; // ?????? `feasible_stations` ?е??


    return selected_station;
}

std::pair<double, int> AGV_solution::findMinEnergyToChargingStation(int v,int j) {
    double min_energy = std::numeric_limits<double>::max();
    Task* task = agvs[v]->task_sequence[j];
    int min_c = -1;
    for (int c = 0; c < pm->C; c++) {
        if (task->cr[c] < min_energy) {
            min_energy = task->cr[c];
            min_c = c;
        }
    }
    return {min_energy, min_c};
}





void AGV_solution::calculateObjectiveValue() {

    OBJ = 0.0;
    double early_time = 0.0;
    double delay_time = 0.0;


    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            double start_time = agv->task_start_times[j];
            if (start_time <= agv->task_sequence[j]->ready_time) {
                early_time += (agv->task_sequence[j]->ready_time - start_time);
            }else{
                delay_time += (start_time - agv->task_sequence[j]->ready_time);
            }
        }
    }

    OBJ = delay_time +  early_time * pm->M;
}

void AGV_solution::printIndex() {
    //??????????
    double total_completion_time = 0.0;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        int num_tasks = agv->task_sequence.size();
        if (num_tasks > 0) {
            total_completion_time += agv->task_end_times[num_tasks - 1];
        }
    }
    cout<<"total_completion_time: "<<total_completion_time<<" ";
    //???????makespan,????????????
    double theoretical_makespan = 0.0;
    vector<double> temp_theoretical_makespan(pm->V, 0.0);
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            Task* task = agv->task_sequence[j];
            temp_theoretical_makespan[v] += task->te + task->tl;
        }
    }
    for (int v = 0; v < pm->V; v++) {
        if (temp_theoretical_makespan[v] > theoretical_makespan) {
            theoretical_makespan = temp_theoretical_makespan[v];
        }
    }
    cout<<"theoretical_makespan: "<<theoretical_makespan<<" ";
    //???????agv???????
    double total_waiting_time = 0.0;
    double max_waiting_time = 0.0;
    int charging_count = 0;

    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                charging_count++;
                double waiting_time = agv->charging_start_times[j] - agv->arrival_at_station[j];
                if (agv->charging_start_times[j] ==0|| agv->arrival_at_station[j] == 0) {
                    cout<<"error waiting time 0"<<endl;
                }
                if (waiting_time<0-1e-3) {
                    cout<<"error waiting time"<<endl;
                    //cout<<"agv: "<<v<<", task: "<<j<<", waiting_time: "<<waiting_time<<endl;
                    system("pause");
                }
                if (waiting_time > 0) {
                    total_waiting_time += waiting_time;

                    if (waiting_time > max_waiting_time) {
                        max_waiting_time = waiting_time;
                    }
                }
            }
        }
    }
    cout<<"????????"<<charging_count<<" ";
    double avg_waiting_time = (charging_count > 0) ? (total_waiting_time / charging_count) : 0.0;

    // ??????
    std::cout << "????????: " << avg_waiting_time <<" ";
    //std::cout << "????????: " << max_waiting_time <<" ";
    // ???????AGV?????????
    std::vector<double> agv_total_waiting_time(pm->V, 0.0);

    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                double waiting_time = agv->charging_start_times[j] - agv->arrival_at_station[j];
                if (waiting_time > 0) {
                    agv_total_waiting_time[v] += waiting_time;
                }
            }
        }
    }

    // ????????AGV????????????
    double avg_total_waiting_time_per_agv = (pm->V > 0) ?
        std::accumulate(agv_total_waiting_time.begin(), agv_total_waiting_time.end(), 0.0) / pm->V : 0.0;

    std::cout << "???AGV??????????: " << avg_total_waiting_time_per_agv << " ";
    // ????????????????????
    std::vector<double> station_occupied_time(pm->C, 0.0);

    for (const auto& agv : agvs) {
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                int cs_id = agv->charging_sessions[j].first;
                double start = agv->charging_start_times[j];
                double end = agv->charging_end_times[j];
                if (start == 0 || end == 0) {
                    cout<<"error charging time negative"<<endl;
                    cout<<"start: "<<start<<", end: "<<end<<endl;
                }
                if (cs_id >= 0 && cs_id < pm->C) {
                    station_occupied_time[cs_id] += (end - start);
                    if (end-start<=0) {
                        cout<<"error charging time"<<endl;
                        cout<<"start: "<<start<<endl;
                        cout<<"end: "<<end<<endl;

                    }
                }
            }
        }
    }

    // ?????????????????
    std::vector<double> station_utilization(pm->C, 0.0);
    for (int c = 0; c < pm->C; ++c) {
        station_utilization[c] = (makespan > 0) ? (station_occupied_time[c] / makespan) : 0.0;
        //std::cout << "???? " << c << " ??????: " << station_utilization[c] << std::endl;
    }

    // ???????????????????????
    double sum_utilization = 0.0;
    double max_utilization = 0.0;
    for (int c = 0; c < pm->C; ++c) {
        sum_utilization += station_utilization[c];
        if (station_utilization[c] > max_utilization) {
            max_utilization = station_utilization[c];
        }
    }
    double avg_utilization = (pm->C > 0) ? (sum_utilization / pm->C) : 0.0;

    std::cout << "??????????: " << avg_utilization <<" ";
    //std::cout << "????????????: " << max_utilization <<" ";

    // ???????AGV???г???????????
    std::vector<double> all_charge_amounts;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                double charge_amount = agv->soc_after_charging[j] - agv->soc_at_cs_arrival[j];
                // if (charge_amount<0-1e-3) {
                //     cout<<"error charge amount negative"<<endl;
                //     cout<<"agv: "<<v<<", task: "<<j<<", charge_amount: "<<charge_amount<<endl;
                //     cout<<"agv->soc_after_charging[j] :"<<agv->soc_after_charging[j] <<endl;
                //     cout<<"agv->soc_at_cs_arrival[j] :"<<agv->soc_at_cs_arrival[j] <<endl;
                //     cout<<"error charge amount"<<endl;
                //     //system("pause");
                // }
                // if (agv->soc_after_charging[j]==0|| agv->soc_at_cs_arrival[j]==0) {
                //     cout<<"error charge amount 0"<<endl;
                // }
                all_charge_amounts.push_back(charge_amount);
            }
        }
    }
    double avg_charge_amount = all_charge_amounts.empty() ? 0.0 :
        std::accumulate(all_charge_amounts.begin(), all_charge_amounts.end(), 0.0) / all_charge_amounts.size();
    double max_charge_amount = all_charge_amounts.empty() ? 0.0 :
        *std::max_element(all_charge_amounts.begin(), all_charge_amounts.end());
    double min_charge_amount = all_charge_amounts.empty() ? 0.0 :
        *std::min_element(all_charge_amounts.begin(), all_charge_amounts.end());
    // ??????
    double var_charge_amount = 0.0;
    if (!all_charge_amounts.empty()) {
        for (double x : all_charge_amounts) {
            var_charge_amount += (x - avg_charge_amount) * (x - avg_charge_amount);
        }
        var_charge_amount /= all_charge_amounts.size();
    }

    std::cout << "??????SOC: " << avg_charge_amount << " ";
    // std::cout << "???γ????: " << max_charge_amount << " ";
    // std::cout << "??С???γ????: " << min_charge_amount << " ";
    std::cout << "?????????: " << var_charge_amount << " ";

    // ???????AGV???г????????????
    std::vector<double> all_charge_times;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                all_charge_times.push_back(agv->charging_durations[j]);
            }
        }
    }
    double avg_charge_time = all_charge_times.empty() ? 0.0 :
        std::accumulate(all_charge_times.begin(), all_charge_times.end(), 0.0) / all_charge_times.size();
    double max_charge_time = all_charge_times.empty() ? 0.0 :
        *std::max_element(all_charge_times.begin(), all_charge_times.end());
    double min_charge_time = all_charge_times.empty() ? 0.0 :
        *std::min_element(all_charge_times.begin(), all_charge_times.end());
    double var_charge_time = 0.0;
    if (!all_charge_times.empty()) {
        for (double t : all_charge_times) {
            var_charge_time += (t - avg_charge_time) * (t - avg_charge_time);
        }
        var_charge_time /= all_charge_times.size();
    }

    std::cout << "?????????: " << avg_charge_time << " ";
    std::cout << "???γ?????: " << max_charge_time << " ";
    std::cout << "??С???γ?????: " << min_charge_time << " ";
    std::cout << "????????: " << var_charge_time << " ";
    // ???????AGV???г?????????????
    std::vector<double> all_charge_start_soc;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                all_charge_start_soc.push_back(agv->soc_at_cs_arrival[j]);
            }
        }
    }
    double avg_charge_start_soc = all_charge_start_soc.empty() ? 0.0 :
        std::accumulate(all_charge_start_soc.begin(), all_charge_start_soc.end(), 0.0) / all_charge_start_soc.size();

    std::cout << "??????SOC: " << avg_charge_start_soc << " ";
    // ???????AGV???г????????????????
    std::vector<double> all_charge_end_soc;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                all_charge_end_soc.push_back(agv->soc_after_charging[j]);
            }
        }
    }
    double avg_charge_end_soc = all_charge_end_soc.empty() ? 0.0 :
        std::accumulate(all_charge_end_soc.begin(), all_charge_end_soc.end(), 0.0) / all_charge_end_soc.size();

    std::cout << "???????SOC: " << avg_charge_end_soc << " ";

    // ??????AGV???????
    std::vector<int> agv_charge_counts(pm->V, 0);
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                agv_charge_counts[v]++;
            }
        }
    }
    double avg_charge_sessions_per_agv = pm->V > 0 ?
        std::accumulate(agv_charge_counts.begin(), agv_charge_counts.end(), 0.0) / pm->V : 0.0;
    int total_charge_sessions = std::accumulate(agv_charge_counts.begin(), agv_charge_counts.end(), 0);
    //std::cout << "???AGV?????????: " << avg_charge_sessions_per_agv << " ";
    std::cout << "????????: " << total_charge_sessions << endl;
    // ??????AGV?????????????????????
    double sum_last_soc = 0.0;
    int agv_count = 0;
    double max_last_soc = 0.0;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        int num_tasks = agv->task_sequence.size();
        if (num_tasks > 0) {
            double last_soc = agv->soc_after_task[num_tasks - 1];
            sum_last_soc += last_soc;
            agv_count++;
            if (agv_count == 1 || last_soc > max_last_soc) {
                max_last_soc = last_soc;
            }
            //std::cout << "AGV " << v << " ??????????????: " << last_soc << std::endl;
        }
    }
    double avg_last_soc = (agv_count > 0) ? (sum_last_soc / agv_count) : 0.0;
    //std::cout << "????AGV?????????????????: " << avg_last_soc <<endl;
    //std::cout << "????AGV?????????????????: " << max_last_soc <<" "<<endl;




}

void AGV_solution::recalculateCompleteSchedule1() {
    double xi_uncertainty = pm->xi_ave_dro;
    double zeta_uncertainty = pm->zeta_cvar_dro;
    cs_available_time.assign(pm->C, 0.0);
    total_waiting_time = 0.0;
    total_charging_time = 0.0;
    total_charging_sessions = 0;
    cs.clear();


    // ????????????????
    std::vector<double> agv_current_time(pm->V, 0.0);
    std::vector<double> agv_current_soc(pm->V);
    std::vector<int> agv_current_task(pm->V, 0);

    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        agv_current_soc[v] = agv->initial_soc;

        int num_tasks = pm->J_v[v];
        if (num_tasks > 0) {
            agv->task_start_times.resize(num_tasks, 0.0);
            agv->task_end_times.resize(num_tasks, 0.0);
            agv->isCharge.resize(num_tasks, false);
            agv->soc_after_task.resize(num_tasks, 0.0);
            agv->soc_before_task.resize(num_tasks, 0.0);
            agv->charging_start_times.resize(num_tasks, 0.0);
            agv->charging_sessions.resize(num_tasks, {-1, -1});
            agv->soc_at_cs_arrival.resize(num_tasks, 0.0);
            agv->arrival_at_station.resize(num_tasks, 0.0);
        }
    }

    // ???????????е???
    while (true) {
        // ????????????????????????δ???????
        int next_agv = -1;
        double earliest_time = std::numeric_limits<double>::max();

        for (int v = 0; v < pm->V; v++) {
            if (agv_current_task[v] < agvs[v]->task_sequence.size()) {
                if (agv_current_time[v] < earliest_time) {
                    earliest_time = agv_current_time[v];
                    next_agv = v;
                }
            }
        }

        // ?????д????????????????
        if (next_agv == -1) break;

        // ??????г???????????
        int v = next_agv;
        int j = agv_current_task[v];
        auto& agv = agvs[v];
        Task* current_task = agv->task_sequence[j];

        // 1. ??????????????????????
        double travel_time_to_task = 0.0;
        double energy_consumption_to_task = 0.0;

        if (j == 0) {
            travel_time_to_task = current_task->te;
            energy_consumption_to_task = current_task->ce;
        } else {
            if (agv->isCharge[j-1]) {
                int charging_station = agv->charging_sessions[j-1].first;
                travel_time_to_task = current_task->tf[charging_station];
                energy_consumption_to_task = current_task->cf[charging_station];
            } else {
                travel_time_to_task = current_task->te;
                energy_consumption_to_task = current_task->ce;
            }
        }

        // 2. ?????????????????
        agv->task_start_times[j] = agv_current_time[v] + travel_time_to_task;
        agv->task_end_times[j] = agv->task_start_times[j] + current_task->tl + xi_uncertainty;

        // 3. ?????????
        agv_current_soc[v] -= energy_consumption_to_task;

        agv->soc_before_task[j] = agv_current_soc[v];
        agv_current_soc[v] -= (current_task->cl + zeta_uncertainty);

        agv->soc_after_task[j] = agv_current_soc[v];

        // 4. ?ж??????????
        if (agv->isCharge[j]) {
            int station_id = agv->charging_sessions[j].first;
            double arrival_at_cs = agv->task_end_times[j] + current_task->tr[station_id];
            double charging_start_time = std::max(arrival_at_cs, cs_available_time[station_id]);
            int soc_start = static_cast<int>(std::round(agv->soc_at_cs_arrival[j]));
            int soc_end = static_cast<int>(std::round(agv->soc_after_charging[j]));
            double charging_duration = pm->get_charge_time_min(soc_start, soc_end) + pm->tu;
//            double charging_duration;
//            if (pm->tm->warm_start) {
//                charging_duration = pm->get_charge_time_min(soc_start, soc_end) + pm->tu;
//            }
//            else {
//                charging_duration = (soc_end-soc_start)/pm->eta + pm->tu;
//            }

            agv->charging_durations[j] = charging_duration;
            double charging_end_time = charging_start_time + charging_duration;
            agv->arrival_at_station[j] = arrival_at_cs;
            agv->soc_at_cs_arrival[j] = agv_current_soc[v] - current_task->cr[station_id];
            if (agv->soc_at_cs_arrival[j]<pm->gamma) {
                cout<<"agv->soc_at_cs_arrival[j]:"<<agv->soc_at_cs_arrival[j]<<endl;
                cout<<"big error6"<<endl;
                system("pause");
            }


            agv->charging_start_times[j] = charging_start_time;
            cs.push_back(std::make_tuple(v, j, charging_start_time, charging_duration));
            total_charging_sessions++;
            total_charging_time += charging_duration ;

            double waiting_time = charging_start_time - arrival_at_cs;
            if (waiting_time > 0) {
                total_waiting_time += waiting_time;
            }

            cs_available_time[station_id] = charging_end_time;
            agv_current_soc[v] = agv->soc_after_charging[j]; // ????????
            agv_current_time[v] = charging_end_time; // ?????????????
        } else {
            agv_current_time[v] = agv->task_end_times[j]; // ?????????????
        }

        // ??????????????
        agv_current_task[v]++;
    }

    // ????makespan
    makespan = 0.0;
    //
    for (int v = 0; v < pm->V; v++) {
        if (!agvs[v]->task_sequence.empty()) {
            makespan = std::max(makespan, agv_current_time[v]);
        }
    }


}




void AGV_solution::resizeVar() {
    getUncertainty();

    // ???ó?????????????????
    cs_available_time.assign(pm->C, 0.0);
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];

        //int num_tasks = agv->task_sequence.size();
        int num_tasks = pm->J_v[v];
        if (num_tasks > 0) {
            agv->task_start_times.resize(num_tasks, 0.0);
            agv->task_end_times.resize(num_tasks, 0.0);
            agv->isCharge.resize(num_tasks, false);
            agv->charging_sessions.resize(num_tasks, {-1, -1});
            agv->charging_start_times.resize(num_tasks, 0.0);
            agv->arrival_at_station.resize(num_tasks, 0.0);
            agv->charging_end_times.resize(num_tasks, 0.0);
            agv->charging_durations.resize(num_tasks, 0.0);

            agv->soc_after_task.resize(num_tasks, 0.0);
            agv->soc_before_task.resize(num_tasks, 0.0);
            agv->soc_at_cs_arrival.resize(num_tasks, 0.0);
            agv->soc_after_charging.resize(num_tasks, 0.0);
            agv->soc_charging_durations.resize(num_tasks, 0.0);
        }
    }
    total_waiting_time = 0.0;
    total_charging_time = 0.0;
    total_charging_sessions = 0;
    cs.clear();
    for (int v=0;v<pm->V;v++) {
        auto& agv = agvs[v];
        int task_id  = pm->J_v[v] - 1;
        double min_energy = std::numeric_limits<double>::max();

        agv->isCharge[task_id] = false;
        agv->charging_sessions[task_id] = {-1, -1};
    }
}

double AGV_solution::simulateTaskExecution(int agv_id, int pos, double prev_energy) {
    auto& agv = agvs[agv_id];
    Task* current_task = agv->task_sequence[pos];
    double energy = prev_energy;

    // ????????????????
    if (pos == 0) {
        energy = agv->initial_soc;
        energy -= current_task->ce;
    } else {
        if (agv->isCharge[pos - 1]) {
            Task* prev_task = agv->task_sequence[pos - 1];
            int station_id = agv->charging_sessions[pos - 1].first;
            //??????????????????????????

            energy = agv->soc_after_charging[pos - 1];

            //??????????
            energy -= current_task->cf[station_id];

        } else {
            energy -= current_task->ce;
        }
    }


    return energy;
}

double AGV_solution::simulateTaskExecutionTime(int agv_id, int pos, double prev_time) {
    auto& agv = agvs[agv_id];
    Task* current_task = agv->task_sequence[pos];
    double time = prev_time;

    // ????????????????
    if (pos == 0) {

        time = current_task->te;
    } else {
        if (agv->isCharge[pos - 1]) {
            Task* prev_task = agv->task_sequence[pos - 1];
            int station_id = agv->charging_sessions[pos - 1].first;

            //??????????
            time += current_task->tf[station_id];

        } else {
            time += current_task->te;
        }
    }


    return time;
}

void AGV_solution::addChargingOperation(int agv_id, int pos, int station_id) {
    auto& agv = agvs[agv_id];

    // ???ó?????
    agv->isCharge[pos] = true;
    agv->charging_sessions[pos] = {station_id, -1};


}
std::size_t AGV_solution::getHash() {
    std::size_t hash = static_cast<std::size_t>(OBJ * 1000);
    return hash;
}

bool AGV_solution::is_feasible() const {


    bool isfeasible = true;
    // 1. ????????????
    for (int v = 0; v < pm->V; v++) {
        const auto& agv = agvs[v];
        for (double soc : agv->soc_after_task) {
            if (soc < pm->gamma - 1e-3) {

                isfeasible =false;
                return false;
            }

        }
    }
    for (int v = 0; v < pm->V; v++) {
        const auto& agv = agvs[v];
        for (double soc : agv->soc_before_task) {

            if (soc < pm->gamma - 1e-3) {
                cout<<"soc:"<<soc<<endl;
                isfeasible =false;
                return false;
            }

        }
    }

    for (int v = 0; v < pm->V; v++) {
        const auto& agv = agvs[v];
        for (int j=0;j<agv->task_sequence.size();j++) {
            if (agv->isCharge[j] && agv->soc_at_cs_arrival[j] < pm->gamma - 1e-3) {
                //cout<<"agv->soc_at_cs_arrival[j] :"<<agv->soc_at_cs_arrival[j] <<endl;
                isfeasible =false;
                return false;
            }
        }

    }


    // 2. ???????????
    for (int c = 0; c < pm->C; c++) {
        std::vector<std::pair<double, double>> intervals;
        for (int v = 0; v < pm->V; v++) {
            const auto& agv = agvs[v];
            for (int j = 0; j < agv->task_sequence.size(); j++) {
                if (agv->isCharge[j] && agv->charging_sessions[j].first == c) {
                    double start = agv->charging_start_times[j];

                    double charging_duration = agv->charging_durations[j];
                    double end = start + charging_duration;
                    intervals.push_back({start, end});
                }
            }
        }

        std::sort(intervals.begin(), intervals.end());
        for (size_t i = 1; i < intervals.size(); i++) {

            if (intervals[i].first < intervals[i-1].second - 1e-3) {
                cout<<"????"<<c<<"?????"<<endl;
                cout<<"intervals[i].first:"<<intervals[i].first<<endl;
                cout<<"intervals[i-1].second:"<<intervals[i-1].second<<endl;
                system("pause");
                isfeasible =false;
                return false;
            }

        }
    }


    // 3. ???????????????е???????
    for (int v = 0; v < pm->V; v++) {
        const auto& agv = agvs[v];
        double last_time = -1e10;
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                double cur_time = agv->charging_start_times[j];
                if (cur_time < last_time - 1e-3) {
                    cout<<"????"<<v<<"??????????????"<<endl;
                    system("pause");
                    isfeasible =false;
                    return false;
                }
                last_time = cur_time;
            }
        }
    }

    //4.????????????????????????
    for (int v = 0; v < pm->V; v++) {
        const auto& agv = agvs[v];
        if (agv->task_sequence.empty()) continue;
        int last_task_index = agv->task_sequence.size() - 1;
        double last_soc = agv->soc_after_task[last_task_index];
        if (abs(last_soc - pm->gamma) > 1e-3) {
            cout<<"????"<<v<<"???????????????????????gamma:"<<last_soc<<endl;
            isfeasible =false;
            return false;
        }
    }



    return isfeasible;
}



// ???ISolution???????

double AGV_solution::getObjectiveValue() {
    return this->OBJ;
}

bool AGV_solution::isFeasible() {

    return is_feasible();
}

// bool AGV_solution::operator<(ISolution& s) {
//     // ????????????????????
//     AGV_solution* other = dynamic_cast<AGV_solution*>(&s);
//     if (other == nullptr) {
//         return false; // ??????????????
//     }
//
//     return true;
// }
bool AGV_solution::operator<(ISolution& s) {
    return static_cast<long long>(1000 * getObjectiveValue()) < static_cast<long long>(1000 * s.getObjectiveValue());
}


ISolution* AGV_solution::getCopy() {
    // ???????????? - ????????????????????
    AGV_solution* copy = new AGV_solution(this->pm, this->solution_name);

    // ?????????????
    copy->OBJ = this->OBJ;
    copy->UB = this->UB;
    copy->LB = this->LB;
    copy->GAP = this->GAP;
    copy->iteration = this->iteration;

    // ??????????
    copy->startTime = this->startTime;
    copy->endTime = this->endTime;
    copy->solve_time = this->solve_time;

    // ?????????????
    copy->zeta_uncertainty = this->zeta_uncertainty;
    copy->xi_uncertainty = this->xi_uncertainty;

    // ???????????????
    copy->total_charging_time = this->total_charging_time;
    copy->total_travel_time = this->total_travel_time;
    copy->total_waiting_time = this->total_waiting_time;
    copy->total_charging_sessions = this->total_charging_sessions;
    copy->makespan = this->makespan;

    // ?????????
    copy->solve_method = this->solve_method;

    // // ????????? - ???????????
    // copy->X_v_j_c = this->X_v_j_c;
    // copy->Y_v1_j1_v2_j2 = this->Y_v1_j1_v2_j2;
    copy->cs_available_time = this->cs_available_time;
    copy->cs = this->cs;

    // ??????????
    copy->LBs = this->LBs;
    copy->UBs = this->UBs;
    copy->iteration_times = this->iteration_times;

    // ????MP???
    copy->x_solution = this->x_solution;
    copy->c_solution = this->c_solution;

    // ????? - ??????shared_ptr??????????Truck????
    copy->agvs = this->agvs;

    return copy;
}

ISolution* AGV_solution::getDeepCopy() {
    AGV_solution* copy = new AGV_solution(this->pm, this->solution_name);

    // ??????????????
    copy->OBJ = this->OBJ;
    copy->UB = this->UB;
    copy->LB = this->LB;
    copy->GAP = this->GAP;
    copy->iteration = this->iteration;

    // ??????
    copy->startTime = this->startTime;
    copy->endTime = this->endTime;
    copy->solve_time = this->solve_time;

    // ??????????
    copy->zeta_uncertainty = this->zeta_uncertainty;
    copy->xi_uncertainty = this->xi_uncertainty;

    // ???????????
    copy->total_charging_time = this->total_charging_time;
    copy->total_travel_time = this->total_travel_time;
    copy->total_waiting_time = this->total_waiting_time;
    copy->total_charging_sessions = this->total_charging_sessions;
    copy->makespan = this->makespan;

    // ?????
    copy->solve_method = this->solve_method;

    // // ???????????
    // copy->X_v_j_c = this->X_v_j_c;
    // copy->Y_v1_j1_v2_j2 = this->Y_v1_j1_v2_j2;
    copy->cs_available_time = this->cs_available_time;
    copy->cs = this->cs;

    // ??????
    copy->LBs = this->LBs;
    copy->UBs = this->UBs;
    copy->iteration_times = this->iteration_times;

    // MP???
    copy->x_solution = this->x_solution;
    copy->c_solution = this->c_solution;

    // ????????????
    copy->agvs.clear();
    for (const auto& agv : this->agvs) {
        auto new_agv = std::make_shared<AGV>(*agv);
        copy->agvs.push_back(new_agv);
    }

    return copy;
}

// ????????????? CSV ???
void AGV_solution::export_to_csv(const std::string& filename) const {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cout << "????????: " << filename << std::endl;
        return;
    }

    // д????
    file << "AGV,Task,ReadyTime,StartTime,EndTime,Delay,StartSOC,EndSOC,Charging\n";

    // д??????????
    for (const auto& agv : agvs) {
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            file << agv->truck_id << ","
                 << j << ","
                 << agv->task_start_times[j] << ","
                 << agv->task_end_times[j] << ","
                 << (j > 0 ? agv->soc_after_task[j-1] : (pm->varphies[agv->truck_id])) << ","
                 << agv->soc_after_task[j] << ","
                 << (agv->isCharge[j] ? "Yes" : "No") << "\n";
        }
    }

    // д???????
    file << "\nCharging Sessions:\n";
    file << "AGV,Task,ChargingStation,Session,StartTime,Duration,SOCBefore,SOCAfter\n";

    for (const auto& agv : agvs) {
        for (int i = 0; i < agv->charging_sessions.size(); i++) {
            int j = 0; // ????????????????
            for (int k = 0; k < agv->isCharge.size(); k++) {
                if (agv->isCharge[k]) {
                    if (i == j) {
                        j = k;
                        break;
                    }
                    j++;
                }
            }

            file << agv->truck_id << ","
                 << j << ","
                 << agv->charging_sessions[i].first << ","
                 << agv->charging_sessions[i].second << ","
                 << agv->charging_start_times[i] << ","
                 << agv->soc_at_cs_arrival[j] << ","
                 << agv->soc_after_charging[j] << "\n";
        }
    }

    file.close();
}

bool AGV_solution::validate_solution() const{

    return true;
}

// Get total charging time across all AGVs
double AGV_solution::get_total_charging_time() const {
    double total = 0.0;
    for (const auto& agv : agvs) {
        total += agv->get_charging_time();
    }
    return total;
}

void AGV_solution::print_solution_summary(bool isPrint) const {

    // ??????????????????????????
    double total_objective = this->total_travel_time + this->total_charging_time + total_waiting_time;

    if (isPrint) {
        // ====== ?????? ======
        std::cout << "\n================ AGV Solution Summary (Time-Cost Objective) ================\n" << std::endl;

        // ??????????????
        std::cout << "--- Solution Info & Resources ---" << std::endl;
        std::cout << std::left << std::setw(30) << "Total AGVs:" << this->pm->V << std::endl;
        std::cout << std::left << std::setw(30) << "Total Tasks:" << this->pm->J << std::endl;
        std::cout << std::left << std::setw(30) << "Total Charging Stations:" << this->pm->C << std::endl;
        std::cout << std::left << std::setw(30) << "theta:" << pm->Theta << std::endl;
        std::cout << std::left << std::setw(30) << "epsilon:" << pm->epsilon << std::endl;

        // ???????
        std::cout << std::left << std::setw(30) << "model_type:";
        switch (this->pm->tm->model_type) {
            case ModelType::DETERMINISTIC:
                std::cout << "DETERMINISTIC (????????)";
                break;
            case ModelType::SAA:
                std::cout << "SAA (???????????)";
                break;
            case ModelType::DRO:
                std::cout << "DRO (?????????)";
                break;
            default:
                std::cout << "UNKNOWN";
                break;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(30) << "Solution Name:" << this->solution_name << std::endl;
        std::cout << std::left << std::setw(30) << "Solve Method:" << this->solve_method << std::endl;

        // ====== ??????????????????????????????======
        std::cout << "\n--- Objective Function Components ---" << std::endl;
        std::cout << std::left << std::setw(30) << "Total Path Time:"
                  << std::fixed << std::setprecision(4) << this->total_travel_time << std::endl;
        std::cout << std::left << std::setw(30) << "Total Charging Time:"
                  << std::fixed << std::setprecision(4) << this->total_charging_time << std::endl;
        std::cout << std::left << std::setw(30) << "Total Waiting Time:"
                  << std::fixed << std::setprecision(4) << total_waiting_time << std::endl;
        std::cout << std::left << std::setw(30) << "Total Objective Value:"
                  << std::fixed << std::setprecision(4) << total_objective << std::endl;

        // ?????????????
        std::cout << std::left << std::setw(30) << "Gurobi OBJ Value:"
                  << std::fixed << std::setprecision(4) << this->OBJ << std::endl;
        double obj_difference = abs(total_objective - this->OBJ);
        if (obj_difference < 1e-4) {
            std::cout << std::left << std::setw(30) << "Objective Consistency:" << "? CONSISTENT" << std::endl;
        } else {
            std::cout << std::left << std::setw(30) << "Objective Consistency:" << "? DIFF=" << obj_difference << std::endl;
        }

        // ???????
        std::cout << "\n--- Performance Metrics ---" << std::endl;
        std::cout << std::left << std::setw(30) << "UB:"
                  << std::fixed << std::setprecision(4) << this->UB << std::endl;
        std::cout << std::left << std::setw(30) << "LB:"
                  << std::fixed << std::setprecision(4) << this->LB << std::endl;
        std::cout << std::left << std::setw(30) << "solve_time:"
                  << std::fixed << std::setprecision(4) << this->solve_time << std::endl;
        std::cout << std::left << std::setw(30) << "GAP:"
                  << std::fixed << std::setprecision(6) << this->GAP << std::endl;

        // ??????????
        std::cout << "\n--- Charging Operations ---" << std::endl;
        std::cout << std::left << std::setw(30) << "Total Charging Sessions:" << total_charging_sessions << std::endl;
        std::cout << std::left << std::setw(30) << "Avg Sessions per AGV:"
                  << std::fixed << std::setprecision(2) << (double)total_charging_sessions / agvs.size() << std::endl;

        if (total_charging_sessions > 0) {
            std::cout << std::left << std::setw(30) << "Avg Waiting per Session:"
                      << std::fixed << std::setprecision(4) << total_waiting_time / total_charging_sessions << std::endl;
            std::cout << std::left << std::setw(30) << "Avg Path Time per Session:"
                      << std::fixed << std::setprecision(4) << this->total_travel_time / total_charging_sessions << std::endl;
        }

        // ???Truck?????
        std::cout << "\n--- Per-AGV Summary ---" << std::endl;
        std::cout << std::left << std::setw(5) << "AGV"
                  << std::setw(8) << "Tasks"
                  << std::setw(12) << "Delay"
                  << std::setw(12) << "Travel"
                  << std::setw(10) << "Min SOC"
                  << std::setw(10) << "Sessions"
                  << std::setw(10) << "Charge%"
                  << "Avg Wait" << std::endl;

        std::cout << std::string(90, '-') << std::endl;

        for (const auto& agv : agvs) {
            // ?????AGV????СSOC
            double agv_min_soc = pm->Pi;
            if (!agv->soc_after_task.empty()) {
                agv_min_soc = *std::min_element(agv->soc_after_task.begin(), agv->soc_after_task.end());
            }

            // ??schedule_events?м???????????????
            int charging_count = 0;
            double agv_total_waiting = 0.0;
            for (const auto& event : agv->schedule_events) {
                if (event.type == ScheduleEventType::CHARGING) {
                    charging_count++;
                    agv_total_waiting += event.waiting_time;
                }
            }

            // ?????????
            double charging_ratio = 0.0;
            if (!agv->task_sequence.empty()) {
                charging_ratio = (double)charging_count / agv->task_sequence.size() * 100.0;
            }

            // ?????????????
            double avg_wait = (charging_count > 0) ? (agv_total_waiting / charging_count) : 0.0;

            std::cout << std::left << std::setw(5) << agv->truck_id
                      << std::setw(8) << agv->task_sequence.size()
                      << std::setw(12) << std::fixed << std::setprecision(2) << agv->total_waiting_time
                      << std::setw(12) << std::fixed << std::setprecision(2) << agv->get_travel_time()
                      << std::setw(10) << std::fixed << std::setprecision(1) << (agv_min_soc / pm->Pi * 100) << "%"
                      << std::setw(10) << charging_count
                      << std::setw(10) << std::fixed << std::setprecision(1) << charging_ratio << "%"
                      << std::fixed << std::setprecision(3) << avg_wait << std::endl;
        }

        // ??????????
        std::cout << "\n--- Charging Station Analysis ---" << std::endl;

        // ??????????????????
        std::map<int, int> station_usage;
        std::map<int, double> station_waiting;

        for (const auto& record : cs) {
            if (std::tuple_size<std::decay<decltype(record)>::type>::value >= 4) {
                try {
                    int station_id = std::get<3>(record);
                    int agv_id = std::get<0>(record);
                    int task_id = std::get<1>(record);
                    double start_time = std::get<2>(record);

                    station_usage[station_id]++;

                    // ?????????????
                    if (agv_id < agvs.size()) {
                        for (const auto& event : agvs[agv_id]->schedule_events) {
                            if (event.type == ScheduleEventType::CHARGING &&
                                event.task_id == task_id &&
                                abs(event.start_time - start_time) < 1e-3) {
                                station_waiting[station_id] += event.waiting_time;
                                break;
                            }
                        }
                    }
                } catch (...) {
                    // ????????????????
                }
            }
        }

        std::cout << std::left << std::setw(10) << "Station"
                  << std::setw(12) << "Sessions"
                  << std::setw(15) << "Total Wait"
                  << std::setw(12) << "Avg Wait"
                  << "Efficiency" << std::endl;
        std::cout << std::string(65, '-') << std::endl;

        for (int i = 0; i < this->pm->C; i++) {
            int sessions = station_usage[i];
            double total_wait = station_waiting[i];
            double avg_wait = (sessions > 0) ? (total_wait / sessions) : 0.0;
            double efficiency = 0.0;

            if (total_wait + sessions * this->pm->tu > 0) {
                efficiency = (sessions * this->pm->tu) / (total_wait + sessions * this->pm->tu) * 100;
            }

            std::cout << std::left << std::setw(10) << ("CS" + std::to_string(i))
                      << std::setw(12) << sessions
                      << std::setw(15) << std::fixed << std::setprecision(3) << total_wait
                      << std::setw(12) << std::fixed << std::setprecision(3) << avg_wait
                      << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
        }

        std::cout << "\n===============================================================================" << std::endl;
    }

    // ====== ???Raw Values??? ======
    std::string model_type_str;
    switch (this->pm->tm->model_type) {
        case ModelType::DETERMINISTIC:
            model_type_str = "DETERMINISTIC";
            break;
        case ModelType::SAA:
            model_type_str = "SAA";
            break;
        case ModelType::DRO:
            model_type_str = "DRO";
            break;
        default:
            model_type_str = "UNKNOWN";
            break;
    }

    // ?????????????????????
    std::cout << this->pm->V << " "
            << this->pm->J << " "
            << this->pm->C << " "
            << std::fixed << std::setprecision(6) << pm->Theta << " "
            << std::fixed << std::setprecision(6) << pm->epsilon << " "
            << model_type_str << " "
            << this->solve_method << " "
            << std::fixed << std::setprecision(6) << total_objective << " "     // ??ü??????????
            << std::fixed << std::setprecision(6) << this->OBJ << " "           // Gurobi????
            << std::fixed << std::setprecision(6) << this->UB << " "
            << std::fixed << std::setprecision(6) << this->LB << " "
            << std::fixed << std::setprecision(4) << this->solve_time << " "
            << std::fixed << std::setprecision(8) << this->GAP << " "
            << std::fixed << std::setprecision(8) << this->total_charging_sessions << " "
            << std::fixed << std::setprecision(6) << this->total_travel_time << " "     // ·?????
            << std::fixed << std::setprecision(6) << this->total_charging_time << " "     // ??????
            << std::fixed << std::setprecision(6) << this->total_waiting_time << " "  // ??????
            << std::endl;

    if (pm->keepAllBounds) {
        for (int i = 0; i < this->iteration; i++) {
            std::cout << this->LBs[i] << " ";
        }
        std::cout << std::endl;

        for (int i = 0; i < this->iteration; i++) {
            std::cout << this->UBs[i] <<  " ";
        }
        std::cout << std::endl;

    }

    if (isPrint) {
        std::cout << "\n===============================================================================" << std::endl;
    }
}
void AGV_solution::printIndex1() {
    //输出总完工时间
    double total_completion_time = 0.0;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        int num_tasks = agv->task_sequence.size();
        if (num_tasks > 0) {
            total_completion_time += agv->task_end_times[num_tasks - 1];
        }
    }
    //cout<<"total_completion_time: "<<total_completion_time<<" ";
    //输出理论makespan,也就是不考虑充电
    double theoretical_makespan = 0.0;
    vector<double> temp_theoretical_makespan(pm->V, 0.0);
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            Task* task = agv->task_sequence[j];
            temp_theoretical_makespan[v] += task->te + task->tl;
        }
    }
    for (int v = 0; v < pm->V; v++) {
        if (temp_theoretical_makespan[v] > theoretical_makespan) {
            theoretical_makespan = temp_theoretical_makespan[v];
        }
    }
    //cout<<"theoretical_makespan: "<<theoretical_makespan<<" ";
    cout<<theoretical_makespan<<" ";
    //计算每个agv的等待时间
    double total_waiting_time = 0.0;
    double max_waiting_time = 0.0;
    int charging_count = 0;

    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                charging_count++;
                double waiting_time = agv->charging_start_times[j] - agv->arrival_at_station[j];
                if (agv->charging_start_times[j] ==0|| agv->arrival_at_station[j] == 0) {
                    cout<<"error waiting time 0"<<endl;
                }
                if (waiting_time<0-1e-3) {
                    cout<<"error waiting time"<<endl;
                    //cout<<"agv: "<<v<<", task: "<<j<<", waiting_time: "<<waiting_time<<endl;
                    system("pause");
                }
                if (waiting_time > 0) {
                    total_waiting_time += waiting_time;

                    if (waiting_time > max_waiting_time) {
                        max_waiting_time = waiting_time;
                    }
                }
            }
        }
    }
    //cout<<"充电次数："<<charging_count<<" ";
    cout<<charging_count<<" ";
    double avg_waiting_time = (charging_count > 0) ? (total_waiting_time / charging_count) : 0.0;

    // 输出结果
    //std::cout << "充电等待时间: " << avg_waiting_time <<" ";
    std::cout << avg_waiting_time <<" ";
    //std::cout << "最大等待时间: " << max_waiting_time <<" ";
    // 计算每个AGV的总等待时间
    std::vector<double> agv_total_waiting_time(pm->V, 0.0);

    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                double waiting_time = agv->charging_start_times[j] - agv->arrival_at_station[j];
                if (waiting_time > 0) {
                    agv_total_waiting_time[v] += waiting_time;
                }
            }
        }
    }

    // 计算所有AGV的平均总等待时间
    double avg_total_waiting_time_per_agv = (pm->V > 0) ?
                                            std::accumulate(agv_total_waiting_time.begin(), agv_total_waiting_time.end(), 0.0) / pm->V : 0.0;

    //std::cout << "每个AGV平均总等待时间: " << avg_total_waiting_time_per_agv << " ";
    std::cout <<  avg_total_waiting_time_per_agv << " ";
    // 统计每个充电站的总占用时间
    std::vector<double> station_occupied_time(pm->C, 0.0);

    for (const auto& agv : agvs) {
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                int cs_id = agv->charging_sessions[j].first;
                double start = agv->charging_start_times[j];
                double end = agv->charging_end_times[j];
                if (start == 0 || end == 0) {
                    cout<<"error charging time negative"<<endl;
                    cout<<"start: "<<start<<", end: "<<end<<endl;
                }
                if (cs_id >= 0 && cs_id < pm->C) {
                    station_occupied_time[cs_id] += (end - start);
                    if (end-start<=0) {
                        cout<<"error charging time"<<endl;
                        cout<<"start: "<<start<<endl;
                        cout<<"end: "<<end<<endl;

                    }
                }
            }
        }
    }

    // 计算每个充电站利用率
    std::vector<double> station_utilization(pm->C, 0.0);
    for (int c = 0; c < pm->C; ++c) {
        station_utilization[c] = (makespan > 0) ? (station_occupied_time[c] / makespan) : 0.0;
        //std::cout << "充电站 " << c << " 利用率: " << station_utilization[c] << std::endl;
    }

    // 计算平均利用率和最大利用率
    double sum_utilization = 0.0;
    double max_utilization = 0.0;
    for (int c = 0; c < pm->C; ++c) {
        sum_utilization += station_utilization[c];
        if (station_utilization[c] > max_utilization) {
            max_utilization = station_utilization[c];
        }
    }
    double avg_utilization = (pm->C > 0) ? (sum_utilization / pm->C) : 0.0;

    //std::cout << "充电站利用率: " << avg_utilization <<" ";
    std::cout << avg_utilization <<" ";
    //std::cout << "最大充电站利用率: " << max_utilization <<" ";

    // 统计所有AGV所有充电操作的充电量
    std::vector<double> all_charge_amounts;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                double charge_amount = agv->soc_after_charging[j] - agv->soc_at_cs_arrival[j];
                // if (charge_amount<0-1e-3) {
                //     cout<<"error charge amount negative"<<endl;
                //     cout<<"agv: "<<v<<", task: "<<j<<", charge_amount: "<<charge_amount<<endl;
                //     cout<<"agv->soc_after_charging[j] :"<<agv->soc_after_charging[j] <<endl;
                //     cout<<"agv->soc_at_cs_arrival[j] :"<<agv->soc_at_cs_arrival[j] <<endl;
                //     cout<<"error charge amount"<<endl;
                //     //system("pause");
                // }
                // if (agv->soc_after_charging[j]==0|| agv->soc_at_cs_arrival[j]==0) {
                //     cout<<"error charge amount 0"<<endl;
                // }
                all_charge_amounts.push_back(charge_amount);
            }
        }
    }
    double avg_charge_amount = all_charge_amounts.empty() ? 0.0 :
                               std::accumulate(all_charge_amounts.begin(), all_charge_amounts.end(), 0.0) / all_charge_amounts.size();
    double max_charge_amount = all_charge_amounts.empty() ? 0.0 :
                               *std::max_element(all_charge_amounts.begin(), all_charge_amounts.end());
    double min_charge_amount = all_charge_amounts.empty() ? 0.0 :
                               *std::min_element(all_charge_amounts.begin(), all_charge_amounts.end());
    // 计算方差
    double var_charge_amount = 0.0;
    if (!all_charge_amounts.empty()) {
        for (double x : all_charge_amounts) {
            var_charge_amount += (x - avg_charge_amount) * (x - avg_charge_amount);
        }
        var_charge_amount /= all_charge_amounts.size();
    }

//    std::cout << "平均充电SOC: " << avg_charge_amount << " ";
//     std::cout << "最大单次充电量: " << max_charge_amount << " ";
//     std::cout << "最小单次充电量: " << min_charge_amount << " ";
//    std::cout << "充电量方差: " << var_charge_amount << " ";
    std::cout << avg_charge_amount << " ";
//    std::cout << max_charge_amount << " ";
//    std::cout << min_charge_amount << " ";
    std::cout << var_charge_amount << " ";

    // 统计所有AGV所有充电操作的充电时间
    std::vector<double> all_charge_times;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                all_charge_times.push_back(agv->charging_durations[j]);
            }
        }
    }
    double avg_charge_time = all_charge_times.empty() ? 0.0 :
                             std::accumulate(all_charge_times.begin(), all_charge_times.end(), 0.0) / all_charge_times.size();
    double max_charge_time = all_charge_times.empty() ? 0.0 :
                             *std::max_element(all_charge_times.begin(), all_charge_times.end());
    double min_charge_time = all_charge_times.empty() ? 0.0 :
                             *std::min_element(all_charge_times.begin(), all_charge_times.end());
    double var_charge_time = 0.0;
    if (!all_charge_times.empty()) {
        for (double t : all_charge_times) {
            var_charge_time += (t - avg_charge_time) * (t - avg_charge_time);
        }
        var_charge_time /= all_charge_times.size();
    }

//    std::cout << "平均充电时间: " << avg_charge_time << " ";
//    std::cout << "最大单次充电时间: " << max_charge_time << " ";
//    std::cout << "最小单次充电时间: " << min_charge_time << " ";
//    std::cout << "充电时间方差: " << var_charge_time << " ";
    std::cout << avg_charge_time << " ";
//    std::cout  << max_charge_time << " ";
//    std::cout  << min_charge_time << " ";
    std::cout << var_charge_time << " ";
    // 统计所有AGV所有充电操作的初始电量
    std::vector<double> all_charge_start_soc;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                all_charge_start_soc.push_back(agv->soc_at_cs_arrival[j]);
            }
        }
    }
    double avg_charge_start_soc = all_charge_start_soc.empty() ? 0.0 :
                                  std::accumulate(all_charge_start_soc.begin(), all_charge_start_soc.end(), 0.0) / all_charge_start_soc.size();

    //std::cout << "平均初始SOC: " << avg_charge_start_soc << " ";
    std::cout << avg_charge_start_soc << " ";
    // 统计所有AGV所有充电操作的充完电后的电量
    std::vector<double> all_charge_end_soc;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                all_charge_end_soc.push_back(agv->soc_after_charging[j]);
            }
        }
    }
    double avg_charge_end_soc = all_charge_end_soc.empty() ? 0.0 :
                                std::accumulate(all_charge_end_soc.begin(), all_charge_end_soc.end(), 0.0) / all_charge_end_soc.size();

    //std::cout << "平均结束SOC: " << avg_charge_end_soc << " ";
    std::cout << avg_charge_end_soc << " ";
    // 统计每个AGV的充电次数
    std::vector<int> agv_charge_counts(pm->V, 0);
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        for (int j = 0; j < agv->task_sequence.size(); j++) {
            if (agv->isCharge[j]) {
                agv_charge_counts[v]++;
            }
        }
    }
    double avg_charge_sessions_per_agv = pm->V > 0 ?
                                         std::accumulate(agv_charge_counts.begin(), agv_charge_counts.end(), 0.0) / pm->V : 0.0;
    int total_charge_sessions = std::accumulate(agv_charge_counts.begin(), agv_charge_counts.end(), 0);
    std::cout << avg_charge_sessions_per_agv << " ";
    //std::cout << "合计充电次数: " << total_charge_sessions << endl;
    // 统计每个AGV最后一个任务完成后的剩余电量
    double sum_last_soc = 0.0;
    int agv_count = 0;
    double max_last_soc = 0.0;
    for (int v = 0; v < pm->V; v++) {
        auto& agv = agvs[v];
        int num_tasks = agv->task_sequence.size();
        if (num_tasks > 0) {
            double last_soc = agv->soc_after_task[num_tasks - 1];
            sum_last_soc += last_soc;
            agv_count++;
            if (agv_count == 1 || last_soc > max_last_soc) {
                max_last_soc = last_soc;
            }
            //std::cout << "AGV " << v << " 最后任务后剩余电量: " << last_soc << std::endl;
        }
    }
    double avg_last_soc = (agv_count > 0) ? (sum_last_soc / agv_count) : 0.0;
    std::cout << avg_last_soc <<endl;
    //std::cout << "所有AGV最后任务后最大剩余电量: " << max_last_soc <<" "<<endl;




}
