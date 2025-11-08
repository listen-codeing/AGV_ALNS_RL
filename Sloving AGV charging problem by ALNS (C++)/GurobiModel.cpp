//
// Created by limin on 8/6/2025.
//

#include "GurobiModel.h"

#include <iomanip>
#include <map>
#include <algorithm>
#include "Model.h"
#include "TestUnit.h"
#include "AGV.h"
#include "Heuristic/AGV_solution.h"


GurobiModel::GurobiModel(Parameter* pm, GurobiModelType model_type, bool isPrint)
        : pm(pm), model_type(model_type), isPrint(isPrint), env(), model(env) {

    switch (pm->tm->model_type) {
        case ModelType::DETERMINISTIC: isDRO = true;
        default: isDRO = false;
    }

    if (isPrint) {
        cout << "创建GurobiModel，类型: 完整模型" << endl;
    }

}

void GurobiModel::build_model() {
    if (isPrint) {
        cout << "构建模型类型: 完整模型" << endl;
    }

    // 统一的建模流程
    create_variables();
    add_constraints();
    set_objective();
    set_parameters();

    model.update();

    write_file();

    if (isPrint) {
        cout << "模型构建完成" << endl;
        cout << "变量数: " << model.get(GRB_IntAttr_NumVars) << endl;
        cout << "约束数: " << model.get(GRB_IntAttr_NumConstrs) << endl;
    }
}

void GurobiModel::write_file() {
    string lp_file_name = "AGV_";

    switch (this->pm->tm->model_type) {
        case ModelType::DETERMINISTIC:
            lp_file_name +=  "det_model.lp";
            break;
        case ModelType::SAA:
            lp_file_name +=  "saa_model.lp";
            break;
        case ModelType::DRO:
            lp_file_name +=  "dro_model.lp";
            break;
        default:
            lp_file_name +=  "det_model.lp";
            break;
    }
    model.write(lp_file_name);

}

void GurobiModel::set_parameters() {

    // 与Python一致的核心参数设置
//    model.set(GRB_DoubleParam_MIPGap, 0.01);  // 设置相对间隙为1%（与Python一致）

    // 设置时间限制（与Python一致）
    model.set(GRB_DoubleParam_TimeLimit, this->pm->time_limit);

}

bool GurobiModel::optimize() {
    // 根据模型类型生成文件名
    string lp_file_name = "AGV_";
    switch (this->pm->tm->model_type) {
        case ModelType::DETERMINISTIC:
            lp_file_name += "det_model.lp";
            break;
        case ModelType::SAA:
            lp_file_name += "saa_model.lp";
            break;
        case ModelType::DRO:
            lp_file_name += "dro_model.lp";
            break;
        default:
            lp_file_name += "unknown_model.lp";
            break;
    }

    // 写入LP文件并更新模型
    model.write(lp_file_name);
    model.update();

    // 设置参数
    model.set(GRB_DoubleParam_MIPGap, 0.01);  // 设置相对间隙为1%
    model.set(GRB_DoubleParam_TimeLimit, this->pm->time_limit);

    // 优化
    model.optimize();
    // // 优化后输出所有设计时间和能量参数
    // std::cout << "\n========= 设计时间和能量参数 =========" << std::endl;
    // for (int v = 0; v < this->pm->V; v++) {
    //     for (int j = 0; j < this->pm->J_v[v]; j++) {
    //         auto task = this->pm->task_sequences[v][j];
    //         std::cout << "AGV " << v << ", 任务 " << j << ":" << std::endl;
    //         std::cout << "  tl(执行时间): " << task->tl << std::endl;
    //         std::cout << "  te(转移时间): " << task->te << std::endl;
    //         std::cout << "  ce(执行能耗): " << task->ce << std::endl;
    //         std::cout << "  cl(转移能耗): " << task->cl << std::endl;
    //         for (int c = 0; c < this->pm->C; c++) {
    //             std::cout << "  tf[" << c << "](充电设置时间): " << task->tf[c] << std::endl;
    //             std::cout << "  tr[" << c << "](到站时间): " << task->tr[c] << std::endl;
    //             std::cout << "  cr[" << c << "](到站能耗): " << task->cr[c] << std::endl;
    //             std::cout << "  cf[" << c << "](充电后能耗): " << task->cf[c] << std::endl;
    //         }
    //     }
    // }
    // std::cout << "初始SOC上界 Pi: " << this->pm->Pi << std::endl;
    // std::cout << "最小SOC下界 gamma: " << this->pm->gamma << std::endl;
    // std::cout << "充电效率 eta: " << this->pm->eta << std::endl;
    // std::cout << "充电设置时间 tu: " << this->pm->tu << std::endl;
    // std::cout << "大M: " << this->pm->M << std::endl;

    // 检查求解状态
    int status = model.get(GRB_IntAttr_Status);
    // 在求解后添加
    // std::cout << "=== 时间变量调试 ===" << std::endl;
    // for (int v = 0; v < this->pm->V; v++) {
    //     for (int j = 0; j < this->pm->J_v[v]; j++) {
    //         std::cout << "AGV " << v << " 任务 " << j << ":" << std::endl;
    //         std::cout << "  开始时间 Ts: " << Ts_v_j[v][j].get(GRB_DoubleAttr_X) << std::endl;
    //         std::cout << "  完成时间 Tc: " << Tc_v_j[v][j].get(GRB_DoubleAttr_X) << std::endl;
    //     }
    // }
    std::cout << "T_max: " << T_max.get(GRB_DoubleAttr_X) << std::endl;

    // 返回是否成功求解
    return (status == GRB_OPTIMAL ||
            status == GRB_SUBOPTIMAL ||
            (status == GRB_TIME_LIMIT && model.get(GRB_IntAttr_SolCount) > 0));
}

void GurobiModel::create_variables() {
    // 为每个AGV分配决策和辅助变量的空间
    X_v_j.resize(this->pm->V);
    Y_v_j_c.resize(this->pm->V);
    Z_v1_j1_v2_j2.resize(this->pm->V);
    Ts_v_j.resize(this->pm->V);
    Tc_v_j.resize(this->pm->V);  // 任务完成时间
    Se_v_j.resize(this->pm->V);
    Sr_v_j.resize(this->pm->V);
    Sf_v_j.resize(this->pm->V);
    tau_v_j.resize(this->pm->V);
    Tr_v_j.resize(this->pm->V);

    for (int v = 0; v < this->pm->V; v++) {
        X_v_j[v].resize(this->pm->J_v[v]);
        Y_v_j_c[v].resize(this->pm->J_v[v]);
        Z_v1_j1_v2_j2[v].resize(this->pm->J_v[v]);
        Ts_v_j[v].resize(this->pm->J_v[v]);
        Tc_v_j[v].resize(this->pm->J_v[v]);
        Se_v_j[v].resize(this->pm->J_v[v]);
        Sr_v_j[v].resize(this->pm->J_v[v]);
        Sf_v_j[v].resize(this->pm->J_v[v]);
        tau_v_j[v].resize(this->pm->J_v[v]);
        Tr_v_j[v].resize(this->pm->J_v[v]);

        for (int j = 0; j < this->pm->J_v[v]; j++) {
            Y_v_j_c[v][j].resize(this->pm->C);
            Z_v1_j1_v2_j2[v][j].resize(this->pm->V);

            for (int vp = 0; vp < this->pm->V; vp++) {
                if(v != vp) {
                    Z_v1_j1_v2_j2[v][j][vp].resize(this->pm->J_v[vp]);
                }
            }
        }
    }

    // 现在根据论文模型创建具体的Gurobi变量
    string name;

    // 创建makespan变量 T_max
    T_max = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "T_max");

    // 创建二元决策变量 x_{v,j}
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            name = "X_" + to_string(v) + "_" + to_string(j);
            X_v_j[v][j] = model.addVar(0, 1, 0, GRB_BINARY, name);
        }
    }

    // 创建三元决策变量 y_{v,j,c}
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            for (int c = 0; c < this->pm->C; c++) {
                name = "Y_" + to_string(v) + "_" + to_string(j) + "_" + to_string(c);
                Y_v_j_c[v][j][c] = model.addVar(0, 1, 0, GRB_BINARY, name);
            }
        }
    }

    // 创建四元决策变量 z_{v,j,v',j'}
    for (int v1 = 0; v1 < this->pm->V; v1++) {
        for (int j1 = 0; j1 < this->pm->J_v[v1]; j1++) {
            for (int v2 = 0; v2 < this->pm->V; v2++) {
                if (v1 != v2) {
                    for (int j2 = 0; j2 < this->pm->J_v[v2]; j2++) {
                        name = "Z_" + to_string(v1) + "_" + to_string(j1) + "_" + to_string(v2) + "_" + to_string(j2);
                        Z_v1_j1_v2_j2[v1][j1][v2][j2] = model.addVar(0, 1, 0, GRB_BINARY, name);
                    }
                }
            }
        }
    }

    // 创建辅助变量
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            // 创建辅助变量 T^s_{v,j}：AGV v第j个任务的开始时间
            name = "Ts_" + to_string(v) + "_" + to_string(j);
            Ts_v_j[v][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);

            // 创建辅助变量 T^c_{v,j}：AGV v第j个任务的完成时间
            name = "Tc_" + to_string(v) + "_" + to_string(j);
            Tc_v_j[v][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);

            // 创建辅助变量 S^e_{v,j}：AGV v完成第j个任务后的SOC
            name = "Se_" + to_string(v) + "_" + to_string(j);
            Se_v_j[v][j] = model.addVar(this->pm->gamma, this->pm->Pi, 0, GRB_CONTINUOUS, name);

            // 创建辅助变量 S^r_{v,j}：AGV v在第j个任务后到达充电站时的SOC
            name = "Sr_" + to_string(v) + "_" + to_string(j);
            Sr_v_j[v][j] = model.addVar(this->pm->gamma, this->pm->Pi, 0, GRB_CONTINUOUS, name);

            // 创建辅助变量 S^f_{v,j}：AGV v在第j个任务后充电完成时的SOC
            name = "Sf_" + to_string(v) + "_" + to_string(j);
            Sf_v_j[v][j] = model.addVar(this->pm->gamma, this->pm->Pi, 0, GRB_CONTINUOUS, name);

            // 创建辅助变量 tau_{v,j}：AGV v在第j个任务后的充电持续时间
            name = "tau_" + to_string(v) + "_" + to_string(j);
            tau_v_j[v][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);

            // 创建辅助变量 T^r_{v,j}：AGV v在第j个任务后到达充电站的时间
            name = "Tr_" + to_string(v) + "_" + to_string(j);
            Tr_v_j[v][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, name);
        }
    }

    // 更新模型以包含新变量
    model.update();
}

void GurobiModel::set_objective() {
    // 根据论文模型设置目标函数：最小化makespan
    GRBLinExpr obj = T_max;
    model.setObjective(obj, GRB_MINIMIZE);

}

// 核心约束添加方法
void GurobiModel::add_constraints() {
    // 约束(eq:det_task_completion): 任务完成时间计算
    if (isPrint) {
        std::cout << "\n约束(eq:det_task_completion): 任务完成时间计算" << std::endl;
    }

    for (int v = 0; v < this->pm->V; v++) {
        string name = "initial_transfer_" + to_string(v);
        model.addConstr(Ts_v_j[v][0] >= this->pm->task_sequences[v][0]->te, name);
    }

    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            string name = "task_completion_" + to_string(v) + "_" + to_string(j);
            if (isPrint) {
                std::cout << "T^c_" << v << "," << j << " = T^s_" << v << "," << j
                          << " + " << this->pm->task_sequences[v][j]->tl << std::endl;
            }
            model.addConstr(Tc_v_j[v][j] == Ts_v_j[v][j] + this->pm->task_sequences[v][j]->tl, name);
        }
    }

    // 约束(eq:det_job_sequence_no_charge): 不充电情况下的任务序列约束
    if (isPrint) {
        std::cout << "\n约束(eq:det_job_sequence_no_charge): 不充电情况下的任务序列" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 1; j < this->pm->J_v[v]; j++) {
            string name = "job_sequence_no_charge_" + to_string(v) + "_" + to_string(j);
            if (isPrint) {
                std::cout << "T^s_" << v << "," << j << " >= T^c_" << v << "," << (j-1)
                          << " + " << this->pm->task_sequences[v][j]->te << std::endl;
            }
            model.addConstr(Ts_v_j[v][j] >= Tc_v_j[v][j-1] + this->pm->task_sequences[v][j]->te, name);
        }
    }

    // 约束(eq:det_job_sequence_with_charge): 充电情况下的任务序列约束
    if (isPrint) {
        std::cout << "\n约束(eq:det_job_sequence_with_charge): 充电情况下的任务序列" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 1; j < this->pm->J_v[v]; j++) {
            string name = "job_sequence_with_charge_" + to_string(v) + "_" + to_string(j);

            // 构建求和表达式
            GRBLinExpr sum_expr = 0;
            for (int c = 0; c < this->pm->C; c++) {
                sum_expr += Y_v_j_c[v][j-1][c] * this->pm->task_sequences[v][j]->tf[c];
            }

            if (isPrint) {
                std::cout << "T^s_" << v << "," << j << " >= T^r_" << v << "," << (j-1)
                          << " + tau_" << v << "," << (j-1) << " + sum(y * tf) - M * (1 - x_"
                          << v << "," << (j-1) << ")" << std::endl;
            }

            model.addConstr(Ts_v_j[v][j] >= Tr_v_j[v][j-1] + tau_v_j[v][j-1] + sum_expr
                                            - this->pm->M * (1 - X_v_j[v][j-1]), name);
        }
    }

    // 约束(eq:det_cs_arrival): AGV到达充电站时间
    if (isPrint) {
        std::cout << "\n约束(eq:det_cs_arrival): AGV到达充电站时间" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            string name = "cs_arrival_" + to_string(v) + "_" + to_string(j);

            // 构建求和表达式
            GRBLinExpr sum_expr = 0;
            for (int c = 0; c < this->pm->C; c++) {
                sum_expr += Y_v_j_c[v][j][c] * this->pm->task_sequences[v][j]->tr[c];
            }

            if (isPrint) {
                std::cout << "T^r_" << v << "," << j << " >= T^c_" << v << "," << j
                          << " + sum(y * tr)" << std::endl;
            }

            model.addConstr(Tr_v_j[v][j] >= Tc_v_j[v][j] + sum_expr, name);
        }
    }

    // 约束(eq:det_makespan): Makespan定义
    if (isPrint) {
        std::cout << "\n约束(eq:det_makespan): Makespan定义" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            string name = "makespan_" + to_string(v) + "_" + to_string(j);
            if (isPrint) {
                std::cout << "T_max >= T^c_" << v << "," << j << std::endl;
            }
            model.addConstr(T_max >= Tc_v_j[v][j], name);
        }
    }

    // 约束(eq:det_charge_binary): 当x_{v,j}=0时，充电时间为0
    if (isPrint) {
        std::cout << "\n约束(eq:det_charge_binary): 充电二进制约束" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            string name = "charge_binary_" + to_string(v) + "_" + to_string(j);
            if (isPrint) {
                std::cout << "tau_" << v << "," << j << " <= x_" << v << "," << j << " * " << this->pm->M << std::endl;
            }
            model.addConstr(tau_v_j[v][j] <= X_v_j[v][j] * this->pm->M, name);
        }
    }

    // 约束(eq:det_charge_duration): 充电时间计算，考虑设置时间和线性充电函数
    if (isPrint) {
        std::cout << "\n约束(eq:det_charge_duration): 充电时间计算" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            string name = "charge_duration_" + to_string(v) + "_" + to_string(j);
            if (isPrint) {
                std::cout << "tau_" << v << "," << j << " >= (1 - x_" << v << "," << j << ") * (-"
                          << this->pm->M << ") + " << this->pm->tu << " + (S^f_" << v << "," << j
                          << " - S^r_" << v << "," << j << ") / " << this->pm->eta << std::endl;
            }
            model.addConstr(tau_v_j[v][j] >= (1 - X_v_j[v][j]) * (-this->pm->M) +
                                             this->pm->tu + (Sf_v_j[v][j] - Sr_v_j[v][j]) / this->pm->eta, name);
        }
    }

    // 约束(eq:det_soc_arrival_cs): AGV到达充电站时的SOC
    if (isPrint) {
        std::cout << "\n约束(eq:det_soc_arrival_cs): AGV到达充电站时的SOC" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            string name = "soc_arrival_cs_" + to_string(v) + "_" + to_string(j);

            // 构建求和表达式
            GRBLinExpr sum_expr = 0;
            for (int c = 0; c < this->pm->C; c++) {
                sum_expr += Y_v_j_c[v][j][c] * this->pm->task_sequences[v][j]->cr[c];
            }

            if (isPrint) {
                std::cout << "S^r_" << v << "," << j << " = S^e_" << v << "," << j
                          << " - sum(y * cr)" << std::endl;
            }

            model.addConstr(Sr_v_j[v][j] == Se_v_j[v][j] - sum_expr, name);
        }
    }

    // 约束(eq:det_soc_after_job_no_charge): 不充电情况下任务后的SOC
    if (isPrint) {
        std::cout << "\n约束(eq:det_soc_after_job_no_charge): 不充电情况下任务后的SOC" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 1; j < this->pm->J_v[v]; j++) {
            string name = "soc_after_job_no_charge_" + to_string(v) + "_" + to_string(j);

            if (isPrint) {
                std::cout << "S^e_" << v << "," << j << " <= S^e_" << v << "," << (j-1)
                          << " - " << this->pm->task_sequences[v][j]->ce
                          << " - " << this->pm->task_sequences[v][j]->cl
                          << " + M * x_" << v << "," << (j-1) << std::endl;
            }

            model.addConstr(Se_v_j[v][j] <= Se_v_j[v][j-1] - this->pm->task_sequences[v][j]->ce
                                            - this->pm->task_sequences[v][j]->cl + this->pm->M * X_v_j[v][j-1], name);
        }
    }

    // 约束(eq:det_soc_after_job_with_charge): 充电情况下任务后的SOC
    if (isPrint) {
        std::cout << "\n约束(eq:det_soc_after_job_with_charge): 充电情况下任务后的SOC" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 1; j < this->pm->J_v[v]; j++) {
            string name = "soc_after_job_with_charge_" + to_string(v) + "_" + to_string(j);

            // 构建求和表达式
            GRBLinExpr sum_expr = 0;
            for (int c = 0; c < this->pm->C; c++) {
                sum_expr += Y_v_j_c[v][j-1][c] * this->pm->task_sequences[v][j]->cf[c];
            }

            if (isPrint) {
                std::cout << "S^e_" << v << "," << j << " <= S^f_" << v << "," << (j-1)
                          << " - sum(y * cf) - " << this->pm->task_sequences[v][j]->cl
                          << " + M * (1 - x_" << v << "," << (j-1) << ")" << std::endl;
            }

            model.addConstr(Se_v_j[v][j] <= Sf_v_j[v][j-1] - sum_expr
                                            - this->pm->task_sequences[v][j]->cl
                                            + this->pm->M * (1 - X_v_j[v][j-1]), name);
        }
    }

    // 约束(eq:det_soc_first_job): 第一个任务后的SOC
    if (isPrint) {
        std::cout << "\n约束(eq:det_soc_first_job): 第一个任务后的SOC" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        string name = "soc_first_job_" + to_string(v);

        if (isPrint) {
            std::cout << "S^e_" << v << ",0 = " << this->pm->varphies[v] << " * " << this->pm->Pi
                      << " - " << this->pm->task_sequences[v][0]->ce
                      << " - " << this->pm->task_sequences[v][0]->cl << std::endl;
        }

        model.addConstr(Se_v_j[v][0] == this->pm->varphies[v] -
                                        this->pm->task_sequences[v][0]->ce - this->pm->task_sequences[v][0]->cl, name);
    }

    // 约束(eq:det_min_soc): 最小SOC约束
    if (isPrint) {
        std::cout << "\n约束(eq:det_min_soc): 最小SOC约束" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            string name = "min_soc_" + to_string(v) + "_" + to_string(j);
            if (isPrint) {
                std::cout << "S^e_" << v << "," << j << " >= " << this->pm->gamma << " * " << this->pm->Pi << std::endl;
            }
            model.addConstr(Se_v_j[v][j] >= this->pm->gamma , name);
        }
    }

    // 约束(eq:det_charge_assignment): 充电决策赋值给充电站
    if (isPrint) {
        std::cout << "\n约束(eq:det_charge_assignment): 充电决策赋值" << std::endl;
    }
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            string name = "charge_assignment_" + to_string(v) + "_" + to_string(j);
            GRBLinExpr sum_expr = 0;

            if (isPrint) {
                std::cout << "sum(y_" << v << "," << j << ",c) = x_" << v << "," << j << std::endl;
            }

            // 构建约束表达式
            for (int c = 0; c < this->pm->C; c++) {
                sum_expr += Y_v_j_c[v][j][c];
            }
            model.addConstr(sum_expr == X_v_j[v][j], name);
        }
    }

    // 约束(Charging-Sequence-1和Charging-Sequence-2): 充电站序列约束
    if (isPrint) {
        std::cout << "\n约束(Charging-Sequence-1): 充电序列约束1" << std::endl;
    }
    for (int v1 = 0; v1 < this->pm->V; v1++) {
        for (int j1 = 0; j1 < this->pm->J_v[v1]; j1++) {
            for (int v2 = v1 + 1; v2 < this->pm->V; v2++) {
                for (int j2 = 0; j2 < this->pm->J_v[v2]; j2++) {
                    for (int c = 0; c < this->pm->C; c++) {
                        string name = "charging_sequence_1_" + to_string(v1) + "_" + to_string(j1) + "_" +
                                      to_string(v2) + "_" + to_string(j2) + "_" + to_string(c);
                        if (isPrint) {
                            std::cout << "T^r_" << v2 << "," << j2 << " >= T^r_" << v1 << "," << j1
                                      << " + tau_" << v1 << "," << j1 << " - M * (3 - z - y1 - y2)" << std::endl;
                        }
                        model.addConstr(Tr_v_j[v2][j2] >= Tr_v_j[v1][j1] + tau_v_j[v1][j1] - this->pm->M *
                                                                                             (3 - Z_v1_j1_v2_j2[v1][j1][v2][j2] - Y_v_j_c[v1][j1][c] - Y_v_j_c[v2][j2][c]), name);
                    }
                }
            }
        }
    }

    if (isPrint) {
        std::cout << "\n约束(Charging-Sequence-2): 充电序列约束2" << std::endl;
    }
    for (int v1 = 0; v1 < this->pm->V; v1++) {
        for (int j1 = 0; j1 < this->pm->J_v[v1]; j1++) {
            for (int v2 = v1 + 1; v2 < this->pm->V; v2++) {
                for (int j2 = 0; j2 < this->pm->J_v[v2]; j2++) {
                    for (int c = 0; c < this->pm->C; c++) {
                        string name = "charging_sequence_2_" + to_string(v1) + "_" + to_string(j1) + "_" +
                                      to_string(v2) + "_" + to_string(j2) + "_" + to_string(c);
                        if (isPrint) {
                            std::cout << "T^r_" << v1 << "," << j1 << " >= T^r_" << v2 << "," << j2
                                      << " + tau_" << v2 << "," << j2 << " - M * (2 + z - y1 - y2)" << std::endl;
                        }
                        model.addConstr(Tr_v_j[v1][j1] >= Tr_v_j[v2][j2] + tau_v_j[v2][j2] - this->pm->M *
                                                                                             (2 + Z_v1_j1_v2_j2[v1][j1][v2][j2] - Y_v_j_c[v1][j1][c] - Y_v_j_c[v2][j2][c]), name);
                    }
                }
            }
        }
    }

    model.update();
}

void GurobiModel::print_result() {
    try {
        // 检查模型状态
        int status = model.get(GRB_IntAttr_Status);

        std::cout << "\n==================== Model Solution Results ====================\n" << std::endl;

        // 显示模型类型
        std::cout << "Model Type: ";
        switch (pm->tm->model_type) {
            case ModelType::DETERMINISTIC:
                std::cout << "Deterministic Model";
                break;
            case ModelType::SAA:
                std::cout << "Sample Average Approximation (SAA)";
                break;
            case ModelType::DRO:
                std::cout << "Distributionally Robust Optimization (DRO)";
                break;
            default:
                std::cout << "Unknown Model Type";
                break;
        }
        std::cout << std::endl;

        // 显示不确定性参数（如果适用）
        if (pm->tm->model_type == ModelType::DRO) {
            std::cout << "DRO Parameters: " << std::endl;
            std::cout << "  Expected Value (sup E[ξ]): " << this->pm->xi_ave_dro << std::endl;
            std::cout << "  CVaR Value (sup CVaR[ξ]): " << this->pm->xi_cvar_dro << std::endl;
            std::cout << "  Ambiguity Set Radius (θ): " << this->pm->Theta << std::endl;
            std::cout << "  Service Level (ε): " << this->pm->epsilon << std::endl;
        } else if (pm->tm->model_type == ModelType::SAA) {
            std::cout << "SAA Parameters: " << std::endl;
            std::cout << "  Expected Value (E[ξ]): " << this->pm->xi_ave_saa << std::endl;
            std::cout << "  Sample Size: " << this->pm->omega_num << std::endl;
        }

        if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL || (status == GRB_TIME_LIMIT && model.get(GRB_IntAttr_SolCount) > 0)) {
            // 模型找到可行解
            std::cout << "Model Status: ";
            switch (status) {
                case GRB_OPTIMAL: std::cout << "OPTIMAL"; break;
                case GRB_SUBOPTIMAL: std::cout << "SUBOPTIMAL"; break;
                case GRB_TIME_LIMIT: std::cout << "TIME_LIMIT (Feasible solution found)"; break;
                default: std::cout << "Other (Status code: " << status << ")"; break;
            }
            std::cout << std::endl;

            // 输出makespan目标函数值
            double makespan = T_max.get(GRB_DoubleAttr_X);
            std::cout << "Objective Value (Makespan): " << makespan << " time units" << std::endl;
            std::cout << "Solve Time: " << model.get(GRB_DoubleAttr_Runtime) << " seconds" << std::endl;
            std::cout << "MIP Gap: " << model.get(GRB_DoubleAttr_MIPGap) << std::endl;

            // 获取不确定性参数
            double uncertainty_time = get_time_uncertainty();

            // 输出详细的调度计划
            std::cout << "\n----- AGV Task Scheduling Plan -----" << std::endl;

            // 调整表头格式
            std::cout << std::left;
            std::cout << std::setw(5) << "AGV"
                      << std::setw(6) << "Task"
                      << std::setw(12) << "Start Time"
                      << std::setw(12) << "End Time"
                      << std::setw(12) << "Task Duration"
                      << std::setw(12) << "Travel Time"
                      << std::setw(12) << "Completion"
                      << std::setw(12) << "Start SOC"
                      << std::setw(12) << "End SOC"
                      << std::setw(8) << "Charge?" << std::endl;

            // 分隔线
            std::cout << std::string(105, '-') << std::endl;

            // 为跟踪SOC计算值
            std::vector<std::vector<double>> calculatedStartSOC(this->pm->V);
            std::vector<std::vector<double>> calculatedEndSOC(this->pm->V);

            // 跟踪每个AGV的最后结束时间
            std::vector<double> lastEndTime(this->pm->V, 0.0);

            for (int v = 0; v < this->pm->V; v++) {
                calculatedStartSOC[v].resize(this->pm->J_v[v]);
                calculatedEndSOC[v].resize(this->pm->J_v[v]);

                // 初始SOC
                double currentSOC = this->pm->varphies[v];

                for (int j = 0; j < this->pm->J_v[v]; j++) {
                    // 从模型获取时间变量
                    double startTime = Ts_v_j[v][j].get(GRB_DoubleAttr_X);
                    double completionTime = Tc_v_j[v][j].get(GRB_DoubleAttr_X);
                    double modelEndSOC = Se_v_j[v][j].get(GRB_DoubleAttr_X);

                    // 计算实际的结束时间（包括不确定性）
                    double actualEndTime = startTime + this->pm->task_sequences[v][j]->tl + uncertainty_time;

                    // 计算SOC（根据模型约束）
                    if (j == 0) {
                        // 第一个任务
                        calculatedStartSOC[v][j] = this->pm->varphies[v] - this->pm->task_sequences[v][j]->ce;
                        calculatedEndSOC[v][j] = calculatedStartSOC[v][j] - this->pm->task_sequences[v][j]->cl;
                    } else {
                        // 后续任务的SOC计算需要考虑是否充电
                        if (j > 0 && X_v_j[v][j-1].get(GRB_DoubleAttr_X) > 0.5) {
                            // 前一个任务后进行了充电
                            double socAfterCharge = Sf_v_j[v][j-1].get(GRB_DoubleAttr_X);
                            // 计算从充电站到当前任务的能耗
                            double travelEnergy = 0.0;
                            for (int c = 0; c < this->pm->C; c++) {
                                if (Y_v_j_c[v][j-1][c].get(GRB_DoubleAttr_X) > 0.5) {
                                    travelEnergy = this->pm->task_sequences[v][j]->cf[c];
                                    break;
                                }
                            }
                            calculatedStartSOC[v][j] = socAfterCharge - travelEnergy;
                        } else {
                            // 直接从前一个任务过来
                            calculatedStartSOC[v][j] = calculatedEndSOC[v][j-1] - this->pm->task_sequences[v][j]->ce;
                        }
                        calculatedEndSOC[v][j] = calculatedStartSOC[v][j] - this->pm->task_sequences[v][j]->cl;
                    }

                    // 是否在此任务后充电
                    bool isCharging = (X_v_j[v][j].get(GRB_DoubleAttr_X) > 0.5);

                    // 更新最后结束时间
                    lastEndTime[v] = actualEndTime;

                    // 恢复右对齐以便数字对齐
                    std::cout << std::right;

                    // 打印行数据
                    std::cout << std::setw(5) << v << std::setw(6) << j
                              << std::setw(12) << std::fixed << std::setprecision(2) << startTime
                              << std::setw(12) << std::fixed << std::setprecision(2) << actualEndTime
                              << std::setw(12) << std::fixed << std::setprecision(2) << this->pm->task_sequences[v][j]->tl
                              << std::setw(12) << std::fixed << std::setprecision(2) << uncertainty_time
                              << std::setw(12) << std::fixed << std::setprecision(2) << completionTime
                              << std::setw(12) << std::fixed << std::setprecision(2) << calculatedStartSOC[v][j]
                              << std::setw(12) << std::fixed << std::setprecision(2) << calculatedEndSOC[v][j]
                              << std::setw(8) << (isCharging ? "Y" : "-") << std::endl;

                    // 如果此任务后需要充电，显示充电信息
                    if (isCharging) {
                        for (int c = 0; c < this->pm->C; c++) {
                            if (Y_v_j_c[v][j][c].get(GRB_DoubleAttr_X) > 0.5) {
                                double chargeArrivalTime = Tr_v_j[v][j].get(GRB_DoubleAttr_X);
                                double chargeTime = tau_v_j[v][j].get(GRB_DoubleAttr_X);
                                double chargeEndTime = chargeArrivalTime + chargeTime;

                                // 计算充电前后的SOC
                                double socBeforeCharge = Sr_v_j[v][j].get(GRB_DoubleAttr_X);
                                double socAfterCharge = Sf_v_j[v][j].get(GRB_DoubleAttr_X);

                                std::cout << std::setw(5) << " " << std::setw(6) << " "
                                          << std::setw(12) << "Charging"
                                          << std::setw(12) << "Station " << c
                                          << std::setw(12) << std::fixed << std::setprecision(2) << chargeTime
                                          << std::setw(12) << std::fixed << std::setprecision(2) << (chargeTime - this->pm->tu)
                                          << std::setw(12) << std::fixed << std::setprecision(2) << chargeEndTime
                                          << std::setw(12) << std::fixed << std::setprecision(2) << socBeforeCharge
                                          << std::setw(12) << std::fixed << std::setprecision(2) << socAfterCharge
                                          << std::endl;
                                break;
                            }
                        }
                    }

                    // 恢复左对齐
                    std::cout << std::left;
                }

                std::cout << "AGV " << v << " Final Completion Time: " << lastEndTime[v] << std::endl;
            }

            // 输出makespan分析
            std::cout << "\n----- Makespan Analysis -----" << std::endl;
            std::cout << "System Makespan: " << makespan << " time units" << std::endl;

            // 找到决定makespan的任务
            double maxCompletionTime = 0.0;
            int criticalAGV = -1, criticalTask = -1;
            for (int v = 0; v < this->pm->V; v++) {
                for (int j = 0; j < this->pm->J_v[v]; j++) {
                    double completionTime = Tc_v_j[v][j].get(GRB_DoubleAttr_X);
                    if (completionTime > maxCompletionTime) {
                        maxCompletionTime = completionTime;
                        criticalAGV = v;
                        criticalTask = j;
                    }
                }
            }

            if (criticalAGV != -1) {
                std::cout << "Critical Path: AGV " << criticalAGV << ", Task " << criticalTask
                          << " (Completion Time: " << maxCompletionTime << ")" << std::endl;
            }

            // 输出充电站使用情况
            std::cout << "\n----- Charging Station Usage -----" << std::endl;
            std::cout << std::left;
            std::cout << std::setw(8) << "Station"
                      << std::setw(8) << "AGV"
                      << std::setw(8) << "Task"
                      << std::setw(12) << "Arrival"
                      << std::setw(12) << "Start Charge"
                      << std::setw(12) << "Charge Time"
                      << std::setw(12) << "End Charge"
                      << std::setw(12) << "Start SOC"
                      << std::setw(12) << "End SOC" << std::endl;

            std::cout << std::string(104, '-') << std::endl;
            std::cout << std::right;

            // 收集所有充电事件
            struct ChargingEvent {
                int stationId, agvId, taskId;
                double arrivalTime, startTime, chargeTime, endTime, startSOC, endSOC;

                ChargingEvent(int s, int a, int t, double arr, double st, double ct, double et, double ss, double es)
                        : stationId(s), agvId(a), taskId(t), arrivalTime(arr), startTime(st),
                          chargeTime(ct), endTime(et), startSOC(ss), endSOC(es) {}
            };

            std::vector<ChargingEvent> allChargingEvents;
            std::vector<bool> stationUsed(this->pm->C, false);

            // 收集所有充电事件
            for (int v = 0; v < this->pm->V; v++) {
                for (int j = 0; j < this->pm->J_v[v]; j++) {
                    if (X_v_j[v][j].get(GRB_DoubleAttr_X) > 0.5) {
                        for (int c = 0; c < this->pm->C; c++) {
                            if (Y_v_j_c[v][j][c].get(GRB_DoubleAttr_X) > 0.5) {
                                stationUsed[c] = true;
                                double arrivalTime = Tr_v_j[v][j].get(GRB_DoubleAttr_X);
                                double chargeTime = tau_v_j[v][j].get(GRB_DoubleAttr_X);
                                double endTime = arrivalTime + chargeTime;
                                double startSOC = Sr_v_j[v][j].get(GRB_DoubleAttr_X);
                                double endSOC = Sf_v_j[v][j].get(GRB_DoubleAttr_X);

                                allChargingEvents.push_back(ChargingEvent(c, v, j, arrivalTime, arrivalTime,
                                                                          chargeTime, endTime, startSOC, endSOC));
                                break;
                            }
                        }
                    }
                }
            }

            // 按充电站ID排序，然后按开始时间排序
            std::sort(allChargingEvents.begin(), allChargingEvents.end(),
                      [](const ChargingEvent& a, const ChargingEvent& b) {
                          if (a.stationId != b.stationId) return a.stationId < b.stationId;
                          return a.startTime < b.startTime;
                      });

            // 输出排序后的充电事件
            int currentStation = -1;
            for (const auto& event : allChargingEvents) {
                if (currentStation != event.stationId) {
                    currentStation = event.stationId;
                    if (currentStation > 0) std::cout << std::string(104, '-') << std::endl;
                    std::cout << "Station " << currentStation << ":" << std::endl;
                }

                std::cout << std::setw(8) << event.stationId
                          << std::setw(8) << event.agvId << std::setw(8) << event.taskId
                          << std::setw(12) << std::fixed << std::setprecision(2) << event.arrivalTime
                          << std::setw(12) << std::fixed << std::setprecision(2) << event.startTime
                          << std::setw(12) << std::fixed << std::setprecision(2) << event.chargeTime
                          << std::setw(12) << std::fixed << std::setprecision(2) << event.endTime
                          << std::setw(12) << std::fixed << std::setprecision(2) << event.startSOC
                          << std::setw(12) << std::fixed << std::setprecision(2) << event.endSOC << std::endl;
            }

            // 显示未使用的充电站
            bool anyUnused = false;
            for (int c = 0; c < this->pm->C; c++) {
                if (!stationUsed[c]) {
                    if (!anyUnused) {
                        std::cout << std::string(104, '-') << std::endl;
                        std::cout << "Unused Stations: ";
                        anyUnused = true;
                    }
                    std::cout << c << " ";
                }
            }
            if (anyUnused) std::cout << std::endl;

            // 输出决策变量总结
            std::cout << "\n----- Decision Variables Summary -----" << std::endl;

            // 充电决策变量
            std::cout << "Charging Decisions (x_{v,j}): ";
            bool anyCharging = false;
            for (int v = 0; v < this->pm->V; v++) {
                for (int j = 0; j < this->pm->J_v[v]; j++) {
                    if (X_v_j[v][j].get(GRB_DoubleAttr_X) > 0.5) {
                        std::cout << "(" << v << "," << j << ") ";
                        anyCharging = true;
                    }
                }
            }
            if (!anyCharging) std::cout << "None";
            std::cout << std::endl;

            // 充电站分配变量
            std::cout << "Station Assignments (y_{v,j,c}): ";
            bool anyAssignment = false;
            for (int v = 0; v < this->pm->V; v++) {
                for (int j = 0; j < this->pm->J_v[v]; j++) {
                    for (int c = 0; c < this->pm->C; c++) {
                        if (Y_v_j_c[v][j][c].get(GRB_DoubleAttr_X) > 0.5) {
                            std::cout << "(" << v << "," << j << "," << c << ") ";
                            anyAssignment = true;
                        }
                    }
                }
            }
            if (!anyAssignment) std::cout << "None";
            std::cout << std::endl;

            // 如果是DRO模型，添加鲁棒性分析
            if (pm->tm->model_type == ModelType::DRO) {
                std::cout << "\n----- Robustness Analysis (DRO-specific) -----" << std::endl;
                std::cout << "DRO Safety Margin: " << (this->pm->xi_cvar_dro - this->pm->xi_ave_saa)
                          << " time units per task" << std::endl;
                std::cout << "This provides protection against " << this->pm->epsilon * 100
                          << "% worst-case scenarios" << std::endl;

                // SOC安全边际分析
                std::cout << "\nSOC Safety Analysis:" << std::endl;
                double minSafetyMargin = std::numeric_limits<double>::max();
                int criticalSOCAGV = -1, criticalSOCTask = -1;

                for (int v = 0; v < this->pm->V; v++) {
                    for (int j = 0; j < this->pm->J_v[v]; j++) {
                        double endSOC = Se_v_j[v][j].get(GRB_DoubleAttr_X);
                        double safetyMargin = endSOC - this->pm->gamma;

                        if (safetyMargin < minSafetyMargin) {
                            minSafetyMargin = safetyMargin;
                            criticalSOCAGV = v;
                            criticalSOCTask = j;
                        }
                    }
                }

                std::cout << "Minimum SOC Safety Margin: " << minSafetyMargin
                          << " (AGV " << criticalSOCAGV << ", Task " << criticalSOCTask << ")" << std::endl;
            }

        } else if (status == GRB_INFEASIBLE) {
            std::cout << "Model is infeasible! Computing conflicting constraints..." << std::endl;

            // 计算IIS
            model.computeIIS();

            std::cout << "\nInfeasible subsystem contains the following constraints:" << std::endl;
            int iis_constrs = 0;
            for (int i = 0; i < model.get(GRB_IntAttr_NumConstrs); i++) {
                GRBConstr constr = model.getConstrs()[i];
                if (constr.get(GRB_IntAttr_IISConstr) == 1) {
                    iis_constrs++;
                    std::cout << "Constraint " << constr.get(GRB_StringAttr_ConstrName) << " is in conflict." << std::endl;
                }
            }
            std::cout << "Total constraints in IIS: " << iis_constrs << std::endl;

            model.write("model_iis.ilp");
            std::cout << "IIS written to 'model_iis.ilp'." << std::endl;

        } else if (status == GRB_UNBOUNDED) {
            std::cout << "Model is unbounded!" << std::endl;
        } else if (status == GRB_TIME_LIMIT && model.get(GRB_IntAttr_SolCount) == 0) {
            std::cout << "Time limit reached, no feasible solution found!" << std::endl;
        } else {
            std::cout << "Optimization terminated with status code: " << status << std::endl;
        }

        std::cout << "\n==================== Results Output Complete ====================\n" << std::endl;

    } catch (GRBException e) {
        std::cout << "Gurobi error code: " << e.getErrorCode() << std::endl;
        std::cout << "Error message: " << e.getMessage() << std::endl;
    } catch (...) {
        std::cout << "Unknown error occurred" << std::endl;
    }
}

std::shared_ptr<AGV_solution> GurobiModel::get_solution() {
    // 检查模型是否已成功求解
    int status = model.get(GRB_IntAttr_Status);
    if (status != GRB_OPTIMAL &&
        status != GRB_SUBOPTIMAL &&
        !(status == GRB_TIME_LIMIT && model.get(GRB_IntAttr_SolCount) > 0)) {
        std::cout << "无法创建解决方案：模型未能达到最优或可行状态。" << std::endl;
        return nullptr;
    }

    // 创建新的解决方案对象
    auto solution = std::make_shared<AGV_solution>(pm, "Gurobi");

    // 设置求解方法
    switch (pm->tm->model_type) {
        case ModelType::DETERMINISTIC:
            solution->solve_method = "Deterministic";
            break;
        case ModelType::SAA:
            solution->solve_method = "SAA";
            break;
        case ModelType::DRO:
            solution->solve_method = "DRO";
            break;
        default:
            solution->solve_method = "Unknown";
            break;
    }

    // 设置求解信息 - 注意：目标函数现在是makespan而不是总延迟
    solution->solve_time = model.get(GRB_DoubleAttr_Runtime);
    solution->OBJ = T_max.get(GRB_DoubleAttr_X);  // makespan值
    solution->UB = model.get(GRB_DoubleAttr_ObjVal);
    solution->LB = model.get(GRB_DoubleAttr_ObjBound);
    solution->GAP = model.get(GRB_DoubleAttr_MIPGap);

    // 清除现有AGV和充电站使用记录
    solution->agvs.clear();
    solution->cs.clear();  // cs是单个vector，直接清空

    // 获取不确定性参数
    double uncertainty_time = get_time_uncertainty();

    // 处理每个AGV
    for (int v = 0; v < this->pm->V; v++) {
        auto agv = std::make_shared<AGV>(*pm, v);

        // 复制任务序列
        agv->task_sequence = this->pm->task_sequences[v];
        int num_tasks = this->pm->J_v[v];

        // 初始化数组
        agv->task_start_times.resize(num_tasks);
        agv->task_end_times.resize(num_tasks);
        agv->isCharge.resize(num_tasks, false);
        agv->soc_after_task.resize(num_tasks);
        agv->soc_at_cs_arrival.resize(num_tasks);
        agv->soc_after_charging.resize(num_tasks);
        agv->charging_sessions.clear();
        agv->charging_start_times.clear();
        agv->charging_durations.clear();

        double totalDelay = 0.0;
        double maxCompletionTime = 0.0;  // 跟踪此AGV的最大完成时间

        // 处理每个任务
        for (int j = 0; j < num_tasks; j++) {
            // 从模型获取时间变量
            double startTime = Ts_v_j[v][j].get(GRB_DoubleAttr_X);
            double completionTime = Tc_v_j[v][j].get(GRB_DoubleAttr_X);

            agv->task_start_times[j] = startTime;

            // 计算实际结束时间（包括不确定性）
            double actualEndTime = startTime + this->pm->task_sequences[v][j]->tl + uncertainty_time;
            agv->task_end_times[j] = actualEndTime;

            // 更新最大完成时间
            maxCompletionTime = std::max(maxCompletionTime, completionTime);

            // 确定此任务后是否充电
            bool isCharging = (X_v_j[v][j].get(GRB_DoubleAttr_X) > 0.5);
            agv->isCharge[j] = isCharging;

            // 从模型获取SOC变量
            double socAfterTask = Se_v_j[v][j].get(GRB_DoubleAttr_X);
            agv->soc_after_task[j] = socAfterTask;

            // 如果任务后需要充电
            if (isCharging) {
                // 查找使用的充电站
                int used_station = -1;
                for (int c = 0; c < this->pm->C; c++) {
                    if (Y_v_j_c[v][j][c].get(GRB_DoubleAttr_X) > 0.5) {
                        used_station = c;
                        break;
                    }
                }

                if (used_station != -1) {
                    // 获取充电相关时间和SOC
                    double arrivalTime = Tr_v_j[v][j].get(GRB_DoubleAttr_X);
                    double socAtArrival = Sr_v_j[v][j].get(GRB_DoubleAttr_X);
                    double chargeDuration = tau_v_j[v][j].get(GRB_DoubleAttr_X);
                    double socAfterCharge = Sf_v_j[v][j].get(GRB_DoubleAttr_X);

                    // 设置AGV的充电信息
                    agv->soc_at_cs_arrival[j] = socAtArrival;
                    agv->soc_after_charging[j] = socAfterCharge;

                    // 记录充电会话信息
                    agv->charging_sessions.push_back(std::make_pair(used_station, 0));
                    agv->charging_start_times.push_back(arrivalTime);
                    agv->charging_durations.push_back(chargeDuration);

                    // 添加到充电站使用记录：(agv_id, task_id, 开始时间, 持续时间)
                    // 注意：您的数据结构最后一个字段是int，所以需要转换duration
                    solution->cs.push_back(
                            std::make_tuple(v, j, arrivalTime, static_cast<int>(chargeDuration)));
                }
            } else {
                // 如果不充电，设置默认值
                agv->soc_at_cs_arrival[j] = 0.0;
                agv->soc_after_charging[j] = 0.0;
            }
        }

        // 将AGV添加到解决方案中
        solution->agvs.push_back(agv);
    }

    // 计算系统级指标
    solution->makespan = T_max.get(GRB_DoubleAttr_X);  // 设置makespan

    // 计算充电站利用率统计
    std::vector<double> station_total_time(this->pm->C, 0.0);

    // 遍历所有充电记录，按充电站统计总充电时间
    for (const auto& record : solution->cs) {
        int station_id = std::get<0>(record);  // 如果第0个字段是station_id
        // 或者如果您的tuple结构不同，可能需要通过其他方式确定station_id
        // 这里假设需要通过agv_id和task_id反向查找station_id

        int agv_id = std::get<0>(record);
        int task_id = std::get<1>(record);
        int duration = std::get<3>(record);

        // 找到这个充电记录对应的充电站
        for (int c = 0; c < this->pm->C; c++) {
            if (Y_v_j_c[agv_id][task_id][c].get(GRB_DoubleAttr_X) > 0.5) {
                station_total_time[c] += duration;
                break;
            }
        }
    }

    // 计算利用率
    for (int c = 0; c < this->pm->C; c++) {
        if (solution->makespan > 0) {
            double utilization = station_total_time[c] / solution->makespan;
            // 如果AGV_solution有charging_station_utilization字段的话
            // solution->charging_station_utilization.push_back(utilization);
        }
    }

    // 计算关键路径信息
    double max_completion = 0.0;
    int critical_agv = -1, critical_task = -1;
    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            double completion = Tc_v_j[v][j].get(GRB_DoubleAttr_X);
            if (completion > max_completion) {
                max_completion = completion;
                critical_agv = v;
                critical_task = j;
            }
        }
    }

    // 计算SOC统计信息
    double min_soc = std::numeric_limits<double>::max();
    double avg_soc = 0.0;
    int soc_count = 0;

    for (int v = 0; v < this->pm->V; v++) {
        for (int j = 0; j < this->pm->J_v[v]; j++) {
            double soc = Se_v_j[v][j].get(GRB_DoubleAttr_X);
            min_soc = std::min(min_soc, soc);
            avg_soc += soc;
            soc_count++;
        }
    }

    double avg_soc_level = (soc_count > 0) ? avg_soc / soc_count : 0.0;
    double soc_safety_margin = min_soc - this->pm->gamma;

    // 计算总充电次数和充电时间
    int total_charging_count = solution->cs.size();  // cs中的记录数就是充电次数
    double total_charging_time = 0.0;
    for (const auto& record : solution->cs) {
        total_charging_time += std::get<3>(record);  // 累加持续时间
    }

    return solution;
}

double GurobiModel::get_time_uncertainty() const {
    switch (pm->tm->model_type) {
        case ModelType::SAA: return pm->xi_ave_saa;
        case ModelType::DRO: return pm->xi_ave_dro;
        default: return 0.0;
    }
}

