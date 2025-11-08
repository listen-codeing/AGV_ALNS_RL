//
// Created by limin on 8/6/2025.
//

#ifndef AGV_SWAPPING_GUROBIMODEL_H
#define AGV_SWAPPING_GUROBIMODEL_H

#include "gurobi_c++.h"
#include <vector>
#include "Heuristic/AGV_solution.h"

using namespace std;

class Parameter;
class AGV_solution;
class AGV;


enum GurobiModelType {
    FULL_MODEL,    // 完整模型（EXACT求解用）
    MP, // 主问题（MP）
    SP     // 子问题（SP）
};

class GurobiModel {

public:
    Parameter *pm = nullptr;

    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    bool isDRO = false;

    GRBQuadExpr obj;

    // 一元变量
    GRBVar T_max;


    // 二元变量
    vector<vector<GRBVar>> X_v_j;
    vector<vector<vector<GRBVar>>> Y_v_j_c;
    vector<vector<vector<vector<GRBVar>>>> Z_v1_j1_v2_j2;

    // 连续时间变量
    vector<vector<GRBVar>> Ts_v_j;
    vector<vector<GRBVar>> Tc_v_j;
    vector<vector<GRBVar>> tau_v_j;
    vector<vector<GRBVar>> Tr_v_j;

    // 连续电量变量
    vector<vector<GRBVar>> Se_v_j;
    vector<vector<GRBVar>> Sr_v_j;
    vector<vector<GRBVar>> Sf_v_j;

    // warm start result
    std::shared_ptr<AGV_solution> latest_solution;
    std::vector<std::vector<bool>> X_result;
    std::vector<std::vector<double>> t_result;
    vector<vector<vector<vector<bool>>>> Y_result;

    bool is_solved = false;

    GurobiModel::GurobiModel(Parameter *pm, GurobiModelType model_type, bool isPrint);

    void build_model();

    bool optimize();

    void print_result();

    shared_ptr<AGV_solution> get_solution();


private:

    GurobiModelType model_type;
    bool isPrint;

    void get_constr();

    void create_variables();

    void add_constraints();

    void set_parameters();

    void write_file();

    void set_objective();

    void print_master_problem_result();

    void print_subproblem_result();


    std::shared_ptr<AGV_solution> mp_solution;  // 保存MP解决方案的引用

    // 路径时间 + 换电时间
    // 等待时间};

    double get_time_uncertainty() const;
};

#endif //AGV_SWAPPING_GUROBIMODEL_H
