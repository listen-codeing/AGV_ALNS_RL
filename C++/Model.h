//
// Created by limin on 13/4/2025.
//

#ifndef QC_AGV_AGV_MODEL_H
#define QC_AGV_AGV_MODEL_H

#include "gurobi_c++.h"
#include <vector>
#include <map>
#include "Heuristic/AGV_solution.h"
#include "GurobiModel.h"

using namespace std;

class Parameter;
class AGV_solution;
class AGV;
class GurobiModel;



class Model {

public:

    Parameter* pm= nullptr;
    bool isPrint = false;

    bool is_solved = true;


    // LBBD算法相关
    std::unique_ptr<GurobiModel> master_model;    // 主问题模型
    std::unique_ptr<GurobiModel> sub_model;       // 子问题模型

    std::shared_ptr<AGV_solution> latest_solution;



    Model(Parameter &truck_pm);

    std::shared_ptr<AGV_solution> Model::heuristic_solve();

    // 主求解方法，返回解决方案
    std::shared_ptr<AGV_solution> solve(SolveMethod method);

    std::shared_ptr<AGV_solution> gurobi_solve();

private:

    double get_time_uncertainty() const;

};


#endif //QC_AGV_AGV_MODEL_H
