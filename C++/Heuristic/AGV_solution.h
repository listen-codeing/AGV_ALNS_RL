//
// Created by limin on 13/4/2025.
//

#ifndef AGV_CHARGING_AGV_SOLUTION_H
#define AGV_CHARGING_AGV_SOLUTION_H

#include <vector>
#include "../Parameter.h"
#include "ISolution.h"
#include <memory>
#include "../Model.h"
#include <random>
#include<chrono>

class AGV;
class Model;

class AGV_solution : public ISolution {

public:
    Parameter* pm = nullptr;                                // �ⲿ����
    std::vector<std::shared_ptr<AGV>> agvs;   // ����AGV�Ľ������

    // ���վʹ�����
    std::vector<std::tuple<int, int, double, double>> cs;
    // cs[c] = ���վ c ��ʹ�ü�¼��ÿ����¼Ϊ (truck_id, task_id, ��ʼʱ��, ����ʱ��)
    std::vector<double> charging_station_utilization ;





    vector<double> cs_available_time;
    double zeta_uncertainty;
    double xi_uncertainty; // ��ȷ���Բ���
    // �������Ԫ����
    std::string solution_name;           // �����������
    double solve_time;                   // ���ʱ�䣨�룩
    std::string solve_method;            // ʹ�õ���ⷽ��
    clock_t startTime;
    clock_t endTime;

    double OBJ = 0.0;
    double total_travel_time = 0.0;
    double total_charging_time = 0.0;
    double total_waiting_time = 0.0;
    int total_charging_sessions = 0;
    double makespan = 0.0;


    // ������ʷ (�����Ҫ��¼)
    vector<double> LBs;         // �½���ʷ
    vector<double> UBs;         // �Ͻ���ʷ
    std::vector<double> iteration_times; // ����ʱ����ʷ

    double UB = 0.0;
    double LB = 0.0;
    double GAP = 0.0;

    int iteration = 0;

    // MP���
    vector<vector<vector<int>>> x_solution;
    vector<vector<double>> c_solution;

    // ���캯��
    AGV_solution(){};
    AGV_solution(Parameter *agv_pm, string model_name);

    // ��������
    void initialize();                            // ��ʼ���������
    bool is_feasible() const;                     // ����������Ƿ����
    bool validate_solution() const;               // ��֤��������Ƿ���������Լ��

    // ʵ��ISolution�ӿڵķ���
    double getObjectiveValue() override;

    bool isFeasible() override;
    bool operator<(ISolution& s) override;
    ISolution* getCopy() override;
    ISolution* getDeepCopy() override;

    // ��������������������
    void print_solution_summary(bool isPrint = false) const;
    void print_detailed_schedule() const;
    void print_charging_schedule() const;
    void export_to_csv(const std::string& filename) const;
    void validate_energy_feasibility() const;

    static std::shared_ptr<AGV_solution> load_from_file(const std::string& filename, Parameter* pm);

    double get_total_charging_time() const;

    double get_objective_value() const;
    double get_worst_case_expected_delay() const;
    void getInitial();
    void getUncertainty();
    double calculateMinEnergyForNextTask(int truck_id,int current_task_idx);
    int selectOptimalChargingStation(int truck_id,int task_idx,double current_soc);
    std::pair<double, int> findMinEnergyToChargingStation(int v,int j);
    void calculateChargingOrderVariables();
    void calculateObjectiveValue();
    std::size_t getHash();
    void recalculateCompleteSchedule1();
    void recalculateChargingOrderVariables();
    void AGV_solution::resizeVar();
    double simulateTaskExecution(int truck_id, int pos, double prev_energy);
    void addChargingOperation(int truck_id, int pos, int station_id);
    bool isLastCharging(int v, int j);
    void printIndex();
    int findNextTruck(const std::vector<double>& agv_current_time,const std::vector<int>& agv_current_task);
    void processTaskWithTimeOptimalCharging( int agv_id, int task_idx,std::vector<double>& agv_current_time,std::vector<double>& agv_current_soc,std::vector<int>& agv_current_task);
    void executeChargingOperation(int agv_id, int task_idx,int station_id, double task_end_time,std::vector<double>& agv_current_time,
                                                               std::vector<double>& agv_current_soc);
    void calculateTaskTiming(int agv_id, int task_idx, double current_time, double &arrival_time,
                             double &task_start_time, double &task_end_time, double &energy_consumption);
    bool mustChargeAfterTask(int agv_id, int task_idx, double current_soc);
    int selectTimeOptimalStation(int agv_id, int task_idx,double current_soc, double task_end_time);
    double calculateTotalChargingTime(int agv_id, int task_idx, int station_id,double task_end_time, double current_soc);
    double simulateTaskExecutionTime(int agv_id, int pos, double prev_time);
    void getInitial1();
    void printIndex1();



};

#endif //AGV_CHARGING_AGV_SOLUTION_H