//
// Created by �Ż� on 2025/8/9.
//

#include "Adaptive_Random_Repair.h"
#include "AGV_solution.h"
#include "../AGV.h"
#include "../Task.h"
#include <algorithm>
#include <limits>
#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>
#include <tuple>

#include "../TestUnit.h"

using namespace std;

// ��ӵ��Ա�־����


Adaptive_Random_Repair::Adaptive_Random_Repair(std::string s)
    : ARepairOperator(s) {
}

Adaptive_Random_Repair::~Adaptive_Random_Repair() {
}

void Adaptive_Random_Repair::repairSolution(ISolution& sol) {
    //cout<<"Adaptive_Random_Repair"<<endl;
    AGV_solution& agv_sol = dynamic_cast<AGV_solution&>(sol);
    agv_sol.resizeVar();


    // Phase 1: Ϊÿ���������м���������޸���ȷ�����λ�úͳ��վ��
    for (int v = 0; v < agv_sol.pm->V; v++) {


        processInterval(agv_sol,v);
    }

    recalculateCompleteSchedule(agv_sol);

    agv_sol.calculateObjectiveValue();




}

void Adaptive_Random_Repair::recalculateCompleteSchedule(AGV_solution& agv_sol){

    agv_sol.cs_available_time.assign(agv_sol.pm->C, 0.0);
    agv_sol.total_waiting_time = 0.0;
    agv_sol.total_charging_time = 0.0;
    agv_sol.total_charging_sessions = 0;
    agv_sol.cs.clear();


    // ��ʼ���¼���������
    std::vector<double> agv_current_time(agv_sol.pm->V, 0.0);
    std::vector<double> agv_current_soc(agv_sol.pm->V);
    std::vector<int> agv_current_task(agv_sol.pm->V, 0);

    for (int v = 0; v < agv_sol.pm->V; v++) {
        auto& agv = agv_sol.agvs[v];
        agv_current_soc[v] = agv->initial_soc;

        int num_tasks = agv_sol.pm->J_v[v];
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

    // �¼������Ĳ��е���
    while (true) {
        // �ҵ���һ��Ҫ������¼��������δ�������
        int next_agv = -1;
        double earliest_time = std::numeric_limits<double>::max();

        for (int v = 0; v < agv_sol.pm->V; v++) {
            if (agv_current_task[v] < agv_sol.agvs[v]->task_sequence.size()) {
                if (agv_current_time[v] < earliest_time) {
                    earliest_time = agv_current_time[v];
                    next_agv = v;
                }
            }
        }

        // ���û�д��������񣬽�������
        if (next_agv == -1) break;

        // ����ѡ�г����ĵ�ǰ����
        int v = next_agv;
        int j = agv_current_task[v];
        auto& agv = agv_sol.agvs[v];
        Task* current_task = agv->task_sequence[j];

        // 1. ���㵽�ﵱǰ��������ʱ����ܺ�
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

        // 2. ��������ʼ�ͽ���ʱ��
        agv->task_start_times[j] = agv_current_time[v] + travel_time_to_task;
        agv->task_end_times[j] = agv->task_start_times[j] + current_task->tl;

        // 3. ���µ���״̬
        agv_current_soc[v] -= energy_consumption_to_task;

        agv->soc_before_task[j] = agv_current_soc[v];
        // if (agv->soc_before_task[j] <agv_sol.pm->gamma-1e-3) {
        //     cout<<"agv->soc_before_task[j]:"<<agv->soc_before_task[j]<<endl;
        //     system("pause");
        // }
        agv_current_soc[v] -= (current_task->cl);

        agv->soc_after_task[j] = agv_current_soc[v];
        // if (agv->soc_after_task[j] <agv_sol.pm->gamma-1e-3) {
        //     cout<<"agv->soc_after_task[j]:"<<agv->soc_after_task[j]<<endl;
        //     system("pause");
        // }

        // 4. �ж��Ƿ���Ҫ���
        if (agv->isCharge[j]) {

            int station_id = agv->charging_sessions[j].first;
            double arrival_at_cs = agv->task_end_times[j] + current_task->tr[station_id];
            double charging_start_time = std::max(arrival_at_cs, agv_sol.cs_available_time[station_id]);
            double charging_end_time = charging_start_time + agv->charging_durations[j];
            agv->arrival_at_station[j] = arrival_at_cs;
            agv->soc_at_cs_arrival[j] = agv_current_soc[v] - current_task->cr[station_id];
            // if (agv->soc_at_cs_arrival[j]<agv_sol.pm->gamma-1e-3) {
            //     cout<<"agv->soc_at_cs_arrival[j]:"<<agv->soc_at_cs_arrival[j]<<endl;
            //     cout<<"big error3"<<endl;
            //     system("pause");
            // }


            agv->charging_start_times[j] = charging_start_time;
            agv->charging_end_times[j] = charging_end_time;
            agv_sol.cs.push_back(std::make_tuple(v, j, charging_start_time, agv->charging_durations[j]));
            agv_sol.total_charging_sessions++;
            agv_sol.total_charging_time += agv->charging_durations[j];

            double waiting_time = charging_start_time - arrival_at_cs;
            if (waiting_time > 0) {
                agv_sol.total_waiting_time += waiting_time;
            }

            agv_sol.cs_available_time[station_id] = charging_end_time;
            agv_current_soc[v] =agv->soc_after_charging[j];
            agv_current_time[v] = charging_end_time; // ���³�����ǰʱ��
        } else {
            agv_current_time[v] = agv->task_end_times[j]; // ���³�����ǰʱ��
        }

        // �ƶ�����һ������
        agv_current_task[v]++;
    }

    // ����makespan
    agv_sol.makespan = 0.0;
    //
    for (int v = 0; v < agv_sol.pm->V; v++) {
        if (!agv_sol.agvs[v]->task_sequence.empty()) {
            agv_sol.makespan = std::max(agv_sol.makespan, agv_current_time[v]);
        }
    }
    agv_sol.calculateObjectiveValue();


}


void Adaptive_Random_Repair::processInterval(AGV_solution& agv_sol, int v) {
    auto& agv = agv_sol.agvs[v];
    //��͵���
    double gamma = agv_sol.pm->gamma;
    //����
    double Pi = agv_sol.pm->Pi;
    int num_tasks = agv->task_sequence.size();
    int last_charge_idx = -1;
    double current_soc = agv->initial_soc;
    Task* current_task = agv->task_sequence[0];



    for (int pos = 0; pos < agv->task_sequence.size()-1; pos++) {

        current_task = agv->task_sequence[pos];
        current_soc = agv_sol.simulateTaskExecution(v, pos, current_soc);
        agv->soc_before_task[pos] = current_soc;
        current_soc -= (current_task->cl);
        agv->soc_after_task[pos] = current_soc;

        bool need_charge = false;
        if (agv->isCharge[pos]) {
            int station_id = agv->charging_sessions[pos].first;
            current_soc -= current_task->cr[station_id];
            agv->soc_at_cs_arrival[pos] = current_soc;

            // if (agv->soc_at_cs_arrival[pos] < gamma-1e-3) {
            //     cout << "here2 error" << endl;
            //     system("pause");
            // }
            current_soc = agv->soc_after_charging[pos];

        }
        else {
            if (pos<agv->soc_before_task.size()-2) {
                // ���㱾����͵���һ������/���վ�����ܺ�

                double next_soc = agv_sol.simulateTaskExecution(v, pos+1, current_soc);

                next_soc -= (agv->task_sequence[pos + 1]->cl);

                if (agv->isCharge[pos+1]) {
                    double min_soc_after_next = next_soc;
                    double min_soc_to_station = agv->task_sequence[pos+1]->cr[agv->charging_sessions[pos + 1].first];

                    min_soc_after_next = min_soc_after_next - min_soc_to_station;

                    if (min_soc_after_next < agv->soc_at_cs_arrival[pos+1]) {



                        need_charge = true;
                    }
                }
                else {
                    // Ԥ�У�������+����һ������/���վ���ܺģ��Ƿ�ȫ�̴��� gamma
                    double min_soc_after_next = next_soc;
                    double min_soc_to_station = std::numeric_limits<double>::max();
                    for (int c = 0; c < agv_sol.pm->C; c++) {
                        double soc_to_station = agv->task_sequence[pos + 1]->cr[c];
                        if (soc_to_station < min_soc_to_station) {
                            min_soc_to_station = soc_to_station;
                        }
                    }

                    min_soc_after_next = min_soc_after_next - min_soc_to_station;

                    if (min_soc_after_next < gamma-1e-3) {


                        need_charge = true;

                    }
                }
            }
            else {
                // ���㱾����͵���һ������/���վ�����ܺ�
                double next_soc = agv_sol.simulateTaskExecution(v, pos+1, current_soc);
                next_soc -= (agv->task_sequence[pos + 1]->cl);
                // Ԥ�У�������+����һ������/���վ���ܺģ��Ƿ�ȫ�̴��� gamma
                double min_soc_after_next = next_soc;

                if (min_soc_after_next < gamma-1e-3) {

                    need_charge = true;
                }
            }

        }
        if (need_charge && !agv->isCharge[pos]) {

            int prev_charge_idx = -1;
            for (int i = pos - 1; i >= 0; --i) {
                if (agv->isCharge[i]) {
                    prev_charge_idx = i;
                    break;
                }
            }
            prev_charge_idx+=1;
            static std::mt19937 rng(std::random_device{}());

            std::vector<int> weights(pos - prev_charge_idx + 1);
            for (int i = 0; i < weights.size(); ++i) {
                weights[i] = i + 1; // Ȩ��Խ����Խ��
            }

            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            int insert_pos = prev_charge_idx + dist(rng);

            pos = insert_pos;


            int station_id = selectOptimalChargingStation(agv_sol, v, pos, agv->soc_after_task[pos]);

            addChargingOperation(agv_sol,v,pos,station_id);

            agv->soc_at_cs_arrival[pos] = agv->soc_after_task[pos]-agv->task_sequence[pos]->cr[station_id];

            // if (agv->soc_at_cs_arrival[pos]<gamma-1e-3) {
            //     cout<<"energy error1"<<endl;
            //     system("pause");
            // }
            agv->isCharge[pos] = true;
            agv->charging_sessions[pos] = {station_id, -1};

            // ������һ������
            int next_charge_pos = -1;
            for (int i = pos + 1; i < agv->task_sequence.size()-1; ++i) {
                if (agv->isCharge[i]) {
                    next_charge_pos = i;
                    break;
                }
            }

            if (next_charge_pos != -1) {

                // 1. �ȼ�������磬ģ���ܷ�ȫ������һ������
                double max_charge = Pi - agv->soc_at_cs_arrival[pos];
                double cur_soc = agv->soc_at_cs_arrival[pos] + max_charge;
                int prev_station = station_id;
                bool feasible = true;

                for (int i = pos + 1; i <= next_charge_pos; ++i) {
                    Task* t = agv->task_sequence[i];
                    if (i == pos + 1) {
                        cur_soc -= t->cf[prev_station];
                    } else {
                        cur_soc -= t->ce;
                    }
                    cur_soc -= t->cl;

                    if (i == next_charge_pos) {
                        int next_station = agv->charging_sessions[next_charge_pos].first;
                        cur_soc -= t->cr[next_station];

                        // �ؼ�������Ӧ�ü���Ƿ��ܴﵽ��һ�������Ԥ�ڵ���
                        if (cur_soc < agv->soc_at_cs_arrival[next_charge_pos]) {
                            feasible = false;
                            break;
                        }
                    } else {
                        // �м�����ֻ��Ҫ��֤����������gamma
                        if (cur_soc < gamma-1e-3) {
                            feasible = false;
                            break;
                        }
                    }
                }
                if (feasible) {

                    // ���Ծ�ȷ�䣬�������ų����
                    double total_consume = 0.0;
                    prev_station = station_id;
                    for (int i = pos + 1; i <= next_charge_pos; ++i) {
                        Task* t = agv->task_sequence[i];
                        if (i == pos + 1) {
                            total_consume += t->cf[prev_station];
                        } else {
                            total_consume += t->ce;
                        }
                        total_consume += t->cl;
                        if (i == next_charge_pos) {
                            int next_station = agv->charging_sessions[next_charge_pos].first;
                            total_consume += t->cr[next_station];
                        }
                    }
                    double required_charge = agv->soc_at_cs_arrival[next_charge_pos] + total_consume - agv->soc_at_cs_arrival[pos];

                    //if (required_charge < 0) required_charge = 0;

                    if (required_charge > max_charge) required_charge = max_charge;

                    agv->soc_charging_durations[pos] = required_charge;
                    agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + required_charge;

                    agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                    if (agv_sol.pm->tm->warm_start) {
//                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//
//                    }
//                    else {
//                        agv->charging_durations[pos] = (agv->soc_after_charging[pos] -agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta + agv_sol.pm->tu;
//                    }
                } else {
                    // ������С�����

                    Task* next_task = agv->task_sequence[pos + 1];
                    double min_soc_to_next_cs = std::numeric_limits<double>::max();
                    if (agv->isCharge[pos + 1]) {
                        int station = agv->charging_sessions[pos + 1].first;
                        min_soc_to_next_cs = agv->task_sequence[pos+1]->cr[station];
                    } else {
                        for (int c = 0; c < agv_sol.pm->C; c++) {
                            double soc_to_station = agv->task_sequence[pos+1]->cr[c];
                            if (soc_to_station < min_soc_to_next_cs) {
                                min_soc_to_next_cs = soc_to_station;
                            }
                        }
                    }

                    double min_charge = next_task->cf[station_id] + next_task->cl + min_soc_to_next_cs + gamma - agv->soc_at_cs_arrival[pos];

                    if (min_charge < 0) min_charge =0.1*max_charge;
                    if (min_charge > max_charge) min_charge = max_charge;
                    if (max_charge-min_charge<1) {
                        agv->soc_charging_durations[pos] = max_charge;
                        agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + max_charge;
                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos]) + agv_sol.pm->tu;
//                        if (agv_sol.pm->tm->warm_start) {
//                            agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos]) + agv_sol.pm->tu;
//
//                        }
//                        else {
//                            agv->charging_durations[pos] = (agv->soc_after_charging[pos] -agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta + agv_sol.pm->tu;
//                        }

                    }
                    else {
                        int n = 20; // �ֶ�����
                        double step = (max_charge - min_charge) / n; // ���㲽��
                        std::vector<double> candidates;
                        // �� min_charge + step ��ʼ���ɺ�ѡֵ
                        for (int i = 3; i <= n; ++i) {
                            candidates.push_back(min_charge + i * step);
                        }
                        // ���û�к�ѡֵ������ʹ�� min_charge + step
                        if (candidates.empty()) {
                            candidates.push_back(min_charge + step);
                        }

                        static std::mt19937 rng(std::random_device{}());
                        std::uniform_int_distribution<int> idx_dist(0, candidates.size() - 1);
                        double random_charge = candidates[idx_dist(rng)];

                        agv->soc_charging_durations[pos] = random_charge;
                        agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + random_charge;

                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos]) + agv_sol.pm->tu;
//                        if (agv_sol.pm->tm->warm_start) {
//                            agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos]) + agv_sol.pm->tu;
//
//                        }
//                        else {
//                            agv->charging_durations[pos] = (agv->soc_after_charging[pos]-agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta + agv_sol.pm->tu;
//                        }
                    }

                }
            }
            //���������Ѿ�û��
            else {

                // 1. ��������磬ģ���ܷ��������ʣ�����������ʣ�������gamma
                double max_charge = Pi - agv->soc_at_cs_arrival[pos];
                double cur_soc = agv->soc_at_cs_arrival[pos] + max_charge;
                int prev_station = station_id;
                bool feasible = true;
                for (int i = pos + 1; i < agv->task_sequence.size(); ++i) {
                    Task* t = agv->task_sequence[i];
                    if (i == pos + 1) {
                        cur_soc -= t->cf[prev_station];
                    } else if (agv->isCharge[i - 1]) {
                        int prev_cs = agv->charging_sessions[i - 1].first;
                        cur_soc -= t->cf[prev_cs];
                    } else {
                        cur_soc -= t->ce;
                    }
                    cur_soc -= t->cl;
                    // û�к������վ������Ҫ�ټ�ȥcr
                    if (cur_soc < gamma-1e-3) {
                        feasible = false;
                        break;
                    }
                }

                // 2. �ж����ʣ�����
                if (feasible && cur_soc >= gamma) {

                    // ���Ծ�ȷ�䣬�������ų����
                    double total_consume = 0.0;
                    prev_station = station_id;
                    for (int i = pos + 1; i < agv->task_sequence.size(); ++i) {
                        Task* t = agv->task_sequence[i];
                        if (i == pos + 1) {
                            total_consume += t->cf[prev_station];
                        } else if (agv->isCharge[i - 1]) {
                            int prev_cs = agv->charging_sessions[i - 1].first;
                            total_consume += t->cf[prev_cs];
                        } else {
                            total_consume += t->ce;
                        }
                        total_consume += t->cl;
                    }

                    double required_charge = gamma + total_consume - agv->soc_at_cs_arrival[pos];

                    //if (required_charge < 0) required_charge = 0;
                    // if (required_charge < 0) {
                    //     agv->isCharge[pos] = false;
                    //     agv->charging_sessions[pos] = {-1, -1};
                    //     continue;
                    //
                    // }
                    if (required_charge > max_charge) required_charge = max_charge;


                    agv->soc_charging_durations[pos] = required_charge;
                    agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + required_charge;

                    agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                    if (agv_sol.pm->tm->warm_start) {
//                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//
//                    }
//                    else {
//                        agv->charging_durations[pos] = (agv->soc_after_charging[pos] - agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta + agv_sol.pm->tu;
//                    }
                    //�������Բ��ñ����ˣ����еĲ����������ˣ�
                    //break;
                } else {

                    // ������С�����
                    Task* next_task = agv->task_sequence[pos + 1];

                    double min_soc_to_next_cs = std::numeric_limits<double>::max();

                    if (agv->isCharge[pos + 1]) {

                        min_soc_to_next_cs = next_task->cr[agv->charging_sessions[pos + 1].first];
                    } else {
                        for (int c = 0; c < agv_sol.pm->C; c++) {

                            double soc_to_station = agv->task_sequence[pos + 1]->cr[c];
                            if (soc_to_station < min_soc_to_next_cs) {
                                min_soc_to_next_cs = soc_to_station;
                            }
                        }
                    }


                    double min_charge = next_task->cf[station_id] + next_task->cl + min_soc_to_next_cs + gamma - agv->soc_at_cs_arrival[pos];


                    if (min_charge < 0) min_charge = 0.1*max_charge;
                    if (min_charge > max_charge) min_charge = max_charge;
                    if (max_charge - min_charge<1) {
                        agv->soc_charging_durations[pos] = max_charge;
                        agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + max_charge;
                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                        if (agv_sol.pm->tm->warm_start) {
//                            agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//
//                        }
//                        else {
//                            agv->charging_durations[pos] = (agv->soc_after_charging[pos] -agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta + agv_sol.pm->tu;
//                        }
                    }
                    else {
                        int n = 20; // �ֶ�����
                        double step = (max_charge - min_charge) / n; // ���㲽��
                        std::vector<double> candidates;

                        // �� min_charge + step ��ʼ���ɺ�ѡֵ
                        for (int i = 3; i <= n; ++i) {
                            candidates.push_back(min_charge + i * step);
                        }

                        // ���û�к�ѡֵ������ʹ�� min_charge + step
                        if (candidates.empty()) {
                            candidates.push_back(min_charge + step);
                        }

                        static std::mt19937 rng(std::random_device{}());
                        std::uniform_int_distribution<int> idx_dist(0, candidates.size() - 1);
                        double random_charge = candidates[idx_dist(rng)];

                        agv->soc_charging_durations[pos] = random_charge;
                        agv->soc_after_charging[pos] = agv->soc_at_cs_arrival[pos] + random_charge;
                        agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                        if (agv_sol.pm->tm->warm_start) {
//                            agv->charging_durations[pos] = agv_sol.pm->get_charge_time_min(agv->soc_at_cs_arrival[pos], agv->soc_after_charging[pos] ) + agv_sol.pm->tu;
//                        }
//                        else {
//                            agv->charging_durations[pos] = (agv->soc_after_charging[pos]-agv->soc_at_cs_arrival[pos])/agv_sol.pm->eta + agv_sol.pm->tu;
//                        }
                    }

                }
            }

        }
    }
}

int Adaptive_Random_Repair::selectOptimalChargingStation(AGV_solution& agv_sol, int agv_id, int pos, double current_soc) {


    auto& agv = agv_sol.agvs[agv_id];
    Task* current_task = agv->task_sequence[pos];
    int num_tasks = agv->task_sequence.size();

    vector<int> feasible_stations;

    // ��������һ������ѡ����еĳ��վ
    // ��ȡ��һ������
    Task* next_task = agv->task_sequence[pos + 1];


    // ����ÿ�����վ�Ŀ�����
    for (int c = 0; c < agv_sol.pm->C; c++) {
        // ����ӳ��վ����һ��������ܺ�
        double energy_to_station = current_task->cr[c];

        double soc_after_charging = current_soc - energy_to_station;


        // ����Ƿ����㰲ȫԣ��
        if (soc_after_charging  > agv_sol.pm->gamma-1e-3) {
            feasible_stations.push_back(c);

        }
    }

    // ���û�п��еĳ��վ�����ص�һ��
    if (feasible_stations.empty()) {
        cout<<"feasible_stations.empty()"<<endl;
        //system("pause");
        return 0;
    }

    // ���ѡ��һ�����еĳ��վ
    static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> dist(0, feasible_stations.size() - 1);
    int selected_station = feasible_stations[dist(rng)];


    return selected_station;
}



void Adaptive_Random_Repair::addChargingOperation(AGV_solution& agv_sol,int agv_id, int pos, int station_id) {
    auto& agv = agv_sol.agvs[agv_id];

    // ���ó�����
    agv->isCharge[pos] = true;
    agv->charging_sessions[pos] = {station_id, -1};

}


