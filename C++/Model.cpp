//
// Created by limin on 13/4/2025.
//

#include "Model.h"
#include <iomanip>
#include <map>
#include <algorithm>
#include "Heuristic/ALNS.h"
#include "TestUnit.h"
#include "AGV.h"
#include "Heuristic/AGV_solution.h"

#include "Heuristic/Charging_Random_Removal.h"
#include "Heuristic/Charging_Critical_Removal.h"

#include "Heuristic/Charging_Worst_Removal.h"
#include "Heuristic/Station_Random_Removal.h"
#include "Heuristic/Station_Critical_Removal.h"

#include "Heuristic/Station_Worst_Removal.h"




#include "Heuristic/ALNS_Parameters.h"
#include "Heuristic/CoolingSchedule_Parameters.h"
#include "Heuristic/ICoolingSchedule.h"
#include "Heuristic/CoolingScheduleFactory.h"
#include "Heuristic/SimulatedAnnealing.h"
#include "Heuristic/OperatorManager.h"
#include "Heuristic/SimpleBestSolutionManager.h"
#include "Heuristic/SimpleLocalSearchManager.h"
#include "Heuristic/ALNS.h"
#include "Heuristic/IBestSolutionManager.h"

#include "Heuristic/Greedy_Best_Repair.h"
#include "Heuristic/Greedy_Random_Repair.h"
#include "Heuristic/Greedy_Time_Repair.h"


#include "Heuristic/Random_Random_Repair.h"
#include "Heuristic/Random_Best_Repair.h"
#include "Heuristic/Random_Time_Repair.h"

#include "Heuristic/Adaptive_Best_Repair.h"
#include "Heuristic/Adaptive_Random_Repair.h"
#include "Heuristic/Adaptive_Time_Repair.h"

#include "Heuristic/NeighSearch_Random_Ls.h"


Model::Model(Parameter &truck_pm)
{
    this->pm = &truck_pm;
    this->isPrint = this->pm->isPrint;
}

std::shared_ptr<AGV_solution> Model::heuristic_solve() {

    // Initialize solution
    auto init_solution = std::make_shared<AGV_solution>(pm, "init_solution");

    init_solution->getInitial1();

    cout<<"V:"<<pm->V<<"C:"<<pm->C<<"J:"<<pm->J;

    if (init_solution->agvs.empty()) {
        cerr << "���󣺳�ʼ������ʧ�ܣ�TrucksΪ��" << endl;
        return nullptr;
    }

    if (isPrint) {
        cout << "��ʼ�����ɳɹ���AGV����: " << init_solution->agvs.size() << endl;
        cout << "��ʼĿ��ֵ: " << init_solution->OBJ << endl;
        cout<<"��ʼ���Ƿ����:"<<init_solution->is_feasible()<<endl;
    }
    ALNS_Parameters alnsParam(pm,init_solution->agvs);
    CoolingSchedule_Parameters csParam(alnsParam); // Cooling schedule parameters

    //ICoolingSchedule* cs = CoolingScheduleFactory::makeCoolingSchedule(dynamic_cast<ISolution&>(*init_solution),csParam);
    ICoolingSchedule* cs = CoolingScheduleFactory::makeCoolingSchedule(
        dynamic_cast<ISolution&>(*init_solution), csParam);
    SimulatedAnnealing sa(*cs);
    OperatorManager opMan(alnsParam);

    Charging_Random_Removal randomR(alnsParam.getMinDestroyPerc(), alnsParam.getMaxDestroyPerc(), "Random Removal");
    opMan.addDestroyOperator(dynamic_cast<ADestroyOperator &>(randomR));

    Charging_Critical_Removal critialR(alnsParam.getMinDestroyPerc(), alnsParam.getMaxDestroyPerc(), "Critial Removal");
    opMan.addDestroyOperator(dynamic_cast<ADestroyOperator &>(critialR));

    Charging_Worst_Removal worstR(alnsParam.getMinDestroyPerc(),alnsParam.getMaxDestroyPerc(), "Critial Removal");
    opMan.addDestroyOperator(dynamic_cast<ADestroyOperator &>(worstR));

    Station_Random_Removal randomS(alnsParam.getMinDestroyPerc(), alnsParam.getMaxDestroyPerc(),"Random Station Removal");
    opMan.addDestroyOperator(dynamic_cast<ADestroyOperator &>(randomS));

    Station_Critical_Removal critialS(alnsParam.getMinDestroyPerc(), alnsParam.getMaxDestroyPerc(), "Critial Station Removal");
    opMan.addDestroyOperator(dynamic_cast<ADestroyOperator &>(critialS));

    Station_Worst_Removal worstS(alnsParam.getMinDestroyPerc(), alnsParam.getMaxDestroyPerc(), "Worst Station Removal");
    opMan.addDestroyOperator(dynamic_cast<ADestroyOperator &>(worstS));






    // //
    Greedy_Random_Repair greedyRandomI("Greedy Random Repair");
    opMan.addRepairOperator(dynamic_cast<ARepairOperator &>(greedyRandomI));
    // //
    // Greedy_Best_Repair greedyBestI("Greedy Best Repair");
    // opMan.addRepairOperator(dynamic_cast<ARepairOperator &>(greedyBestI));
    //
    Greedy_Time_Repair greedyTimeI("Greedy Time Repair");
    opMan.addRepairOperator(dynamic_cast<ARepairOperator &>(greedyTimeI));
    //
    //
    //
    //
    Random_Random_Repair randomRandomI("Random Random Repair");
    opMan.addRepairOperator(dynamic_cast<ARepairOperator &>(randomRandomI));
    // // // //
    //  Random_Best_Repair randomBestI("Random Best Repair");
    //  opMan.addRepairOperator(dynamic_cast<ARepairOperator &>(randomBestI));
    //
    Random_Time_Repair randomTimeI("Random Time Repair");
    opMan.addRepairOperator(dynamic_cast<ARepairOperator &>(randomTimeI));
    // //
    // Adaptive_Best_Repair adaptiveBestI("Adaptive Best Repair");
    // opMan.addRepairOperator(dynamic_cast<ARepairOperator &>(adaptiveBestI));
    //
    Adaptive_Random_Repair adaptiveRandomI("Adaptive Random Repair");
    opMan.addRepairOperator(dynamic_cast<ARepairOperator &>(adaptiveRandomI));

    Adaptive_Time_Repair adaptiveTimeI("Adaptive Time Repair");
    opMan.addRepairOperator(dynamic_cast<ARepairOperator &>(adaptiveTimeI));




    SimpleBestSolutionManager bestSM(alnsParam);
    SimpleLocalSearchManager simpleLsManager(alnsParam);
    NeighSearch_Random_Ls neighSearchRandomLs("AGV Neigh Search Random Ls");
    simpleLsManager.addLocalSearchOperator(neighSearchRandomLs);

    ALNS alns("TRUCKRobust", //
            dynamic_cast<ISolution&>(*init_solution), //
            dynamic_cast<IAcceptanceModule&>(sa), //
            alnsParam, //
            dynamic_cast<AOperatorManager&>(opMan), //
            dynamic_cast<IBestSolutionManager&>(bestSM),
            dynamic_cast<ILocalSearchManager&>(simpleLsManager));

    alns.solve();

    AGV_solution* agvSol = dynamic_cast<AGV_solution*>(alns.getBestSolutionManager()->getBestSol());

    cout<<pm->J<<pm->C<<pm->V;
    cout << "��ʼĿ��ֵ: " << init_solution->OBJ  <<" "<<"agvSol:"<<agvSol->OBJ<<" ";
    //cout<<"agvSol->total_waiting_time:"<<agvSol->total_waiting_time<<endl;
    agvSol->printIndex1();


     init_solution = nullptr;
    // Update objective value
    return init_solution;
}

std::shared_ptr<AGV_solution> Model::gurobi_solve() {
    if (isPrint) {
        cout << "\n========== ��ȷ��⿪ʼ ==========" << endl;
        cout << "�����ģ: AGV��=" << pm->V << ", ��������="
             << accumulate(pm->J_v.begin(), pm->J_v.end(), 0) << endl;
    }

    try {
        // ��������ģ��
        auto gurobi_model = std::make_unique<GurobiModel>(pm, FULL_MODEL, isPrint);

        // ����ģ��
        gurobi_model->build_model();

        if (isPrint) {
            cout << "ģ�͹�����ɣ���ʼ�Ż�..." << endl;
        }

        // ���ģ��
        if (!gurobi_model->optimize()) {
            cerr << "Gurobiģ�����ʧ��!" << endl;
            return nullptr;
        }

        // ��ӡ��ϸ����������Ҫ��
        if (isPrint) {
            gurobi_model->print_result();
        }

        latest_solution = gurobi_model->get_solution();

        if (!latest_solution) {
            cerr << "����ȡʧ��!" << endl;
            return nullptr;
        }

        // ������ⷽ����ʶ
        latest_solution->solve_method = "EXACT_GUROBI";
        switch (pm->tm->model_type) {
            case ModelType::DETERMINISTIC:
                latest_solution->solve_method += "_DETERMINISTIC";
                break;
            case ModelType::SAA:
                latest_solution->solve_method += "_SAA";
                break;
            case ModelType::DRO:
                latest_solution->solve_method += "_DRO";
                break;
            default:
                latest_solution->solve_method += "_DETERMINISTIC";
                break;
        }

        if (isPrint) {
            cout << "========== ��ȷ������ ==========" << endl;
            cout << "����Ŀ��ֵ: " << latest_solution->OBJ << endl;
        }

        return latest_solution;

    } catch (const std::exception& e) {
        cerr << "��ȷ�������з����쳣: " << e.what() << endl;
        return nullptr;
    } catch (...) {
        cerr << "��ȷ�������з���δ֪�쳣" << endl;
        return nullptr;
    }
}


// Implementation in Model.cpp
std::shared_ptr<AGV_solution> Model::solve(SolveMethod method) {
    clock_t start_time = clock(); // ��¼��ʼʱ��

    switch (method) {
        case GUROBI:
            latest_solution = gurobi_solve();
            break;
        case HEURISTICS:
            latest_solution = heuristic_solve();

    }

    is_solved = true;
    // �������ʱ��
    clock_t end_time = clock();
    double solve_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // ���ý�����������ʱ��
    if (latest_solution) {
//        latest_solution->solve_time = solve_time;
        latest_solution->print_solution_summary(isPrint);
    }

    return latest_solution;
}

// ��ӻ�ȡʱ�䲻ȷ���Եĸ���������GurobiModel��
double Model::get_time_uncertainty() const {
    switch (pm->tm->model_type) {
        case ModelType::SAA: return pm->xi_ave_saa;
        case ModelType::DRO: return pm->xi_ave_dro;
        default: return 0.0;
    }
}

