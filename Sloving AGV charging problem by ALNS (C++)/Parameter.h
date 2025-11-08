//
// Created by limin on 13/4/2025.
//

#ifndef AGV_CHARGING_PARAMETER_H
#define AGV_CHARGING_PARAMETER_H

#include "gurobi_c++.h"
#include "Task.h"
#include <string>
#include <unordered_map>



class TestUnit;
class TestCase;

using namespace std;

enum ModelType {
    DETERMINISTIC,  // 确定性模型
    SAA,           // SAA模型
    DRO,            // DRO模型
};

enum SolveMethod {
    GUROBI,      // 使用精确优化（Gurobi）
    HEURISTICS, // 基于ALNS求解
    LBBD_GUROBI,
    LBBD_RULE,  //
};

enum class SubproblemSolveMethod {
    GUROBI_MODEL,     // 使用Gurobi模型求解
    RULE_BASED        // 使用基于规则的方法求解
};

struct ChargeData {
    int no;
    double soc_start;
    double soc_end;
    double soc_delta;
    double time_min;
    double time_hour;
};


class Parameter {
public:
    TestUnit* tm;
    std::vector<std::vector<double>> distance; // 距离矩阵
    std::vector<std::vector<std::pair<int, int>>> cs_schedule; // cs_schedule[c] = [(truck_id, task_id), ...]
    std::vector<double> cs_available_time; // 每个充电站的可用时间
    int V;  // AGV数量
    int J;  // 集装箱总数
    int C;  // 充电站数量

    vector<int> J_v;  // 每个AGV任务数量
    vector<double> varphies;  // 初始SOC比率

    double V_loading = 3 * 0.06; // km/min
    double V_unloading = 5 * 0.06; // km/min

    // 电池参数
    // double Delta = 0.333; // kWh/min
    // double delta = 0.267; // kWh/min
    double Delta = 1.7625*6/2.83; // kWh/km
    double delta = 1.7625*5.0/2.83; // kWh/km
    double eta = 4; // 充电站充电率 %
    double tu = 5; // 固定设置时间
    double Pi = 100; // 最大电池电量
    double gamma_percent = 0.10;  // 最小电池电量比率
    double gamma = Pi * gamma_percent;  // 最小电池电量比率

    std::vector<ChargeData> charge_table;
    void read_charge_table_from_csv();
    double get_charge_time_min(double true_soc_start, double true_soc_end);
    double get_max_charging_energy(double true_soc_start, double true_soc_end, double max_charge);
    double get_best_soc_delta(double soc_start);
    // 不确定性参数
    double omega_num = 200;    // Wasserstein相关临时变量
    double epsilon = 0.2;  // 服务水平，los = (1-epsilon)
    double Theta = 0.1;    // Wasserstein距离参数
    double wassterTemp;    // Wasserstein相关临时变量

    // 大M值
    double M = 1000; // 大M值
    double time_limit = 3600;

    // 不确定参数
    vector<double> time_tilde;  // 随机项
    double xi_ave_saa;  // SAA平均值
    double xi_ave_dro;  // Wasserstein距离下的平均值
    double xi_cvar_dro; // Wasserstein距离下的CVaR值
    double threshold;    // Wasserstein相关临时变量
    double maxIndex;    // Wasserstein相关临时变量
    double minIndex;    // Wasserstein相关临时变量

    // 新增：能量不确定性相关变量
    vector<double> zeta_tilde;           // 能量不确定项 (对应论文中的 ζ)
    double zeta_ave_saa;          // 能量不确定性SAA平均值
    double zeta_ave_dro;          // 能量不确定性DRO平均值
    double zeta_cvar_dro;         // 能量不确定性DRO的CVaR值

    bool isPrint = false;
    bool useCSV = true;    //
    bool keepAllBounds = true;
    // 其他参数
    int max_time = 3600;   // 最大求解时间
    double tolerance = 0.0000000001; // LBBD_GUROBI 阈值
    int max_iterations = 10e6; // LBBD_GUROBI 最大迭代次数

    string fileName;

    Parameter(TestUnit *tm, TestCase testCase);

    void generate_distance_matrix();

    void generate_agv_container_num();
    void generate_agv_init_SOC();


    std::unordered_map<int, Task> tasks;  // 任务字典，使用任务ID作为键
    std::vector<std::vector<Task*>> task_sequences;  // 使用指针指向tasks中的任务
    void reset();

    void generate_tasks();             // 生成任务
    // 将任务分配给AGVs

    void print_tasks();
    void save_tasks_to_file(const string &filename);
    void assign_tasks_to_trucks();
    void generate_all_task();

    void generate_travel_time_scenarios();
    double calculate_time_ave_saa();  // 计算Wasserstein距离下的平均值
    double calculate_time_ave_dro();  // 计算Wasserstein距离下的平均值
    double calculate_time_cvar_dro(); // 计算Wasserstein距离下的CVaR
    void calculate_time_dro();

    // 新增方法声明
    void generate_energy_uncertainty_scenarios();  // 生成能量不确定项
    double calculate_zeta_ave_saa();               // 计算能量不确定性SAA平均值
    double calculate_zeta_ave_dro();               // 计算能量不确定性DRO平均值
    double calculate_zeta_cvar_dro();              // 计算能量不确定性DRO的CVaR
    void calculate_zeta_dro();                     // 计算所有能量DRO相关值

    void process_precision();
    void adjustFloatPrecision(double &value);

    void read_distance_from_csv();

    void amplify_critical_distances();

    void analyze_distance_characteristics();
};


#endif //AGV_CHARGING_PARAMETER_H
