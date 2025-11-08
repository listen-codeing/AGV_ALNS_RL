//
// Created by limin on 2023/10/28.
//

#include "TestUnit.h"
#include "Parameter.h"
#include <iostream>
#include <sstream>
#include "Heuristic/AGV_solution.h"
#include "Model.h"
#include <algorithm>
#include "random_seed.h"

TestCase::TestCase(int container_num, int truck_num, int charge_station_num) {
    this->container_num = container_num;
    this->truck_num = truck_num;
    this->charge_station_num = charge_station_num;
}

void TestCase::print_begin(){
    std::cout << "****************************" << std::endl;
    std::cout << " container_num: " << container_num << "truck_num: " << truck_num << std::endl;
}

TestUnit::TestUnit(std::string testType, SolveMethod solve_method, bool warm_start, ModelType model_type, bool Totxt) {

    this->testType = testType;
    this->Totxt = Totxt;
    this->solve_method = solve_method;
    this->warm_start = warm_start;
    this->model_type = model_type;
    get_test_data();
    get_fileName();
};

void TestUnit::run() {
    // ��������ض���
    OutputRedirector redirector(Totxt, fileName_printOut);

    // �������в�������
    for (auto& testCase : testCases) {
        // ��֤ÿ�������������һ��
        //sharedGenerator.seed(1);
        // ��ʼ�����Ի���
        Parameter pm(this, testCase);

        print_begin(pm);


        // ����ģ��
        Model truck_DRO(pm);

        auto solution = truck_DRO.solve(this->solve_method);
        // auto solution1 = truck_DRO.solve(this->solve_method);
        // auto solution2 = truck_DRO.solve(this->solve_method);
        // auto solution3 = truck_DRO.solve(this->solve_method);
        // auto solution4 = truck_DRO.solve(this->solve_method);
        // auto solution5 = truck_DRO.solve(this->solve_method);

//        // ����ԭʼģ������״̬
//        ModelType originalModelType = pm.tm->model_type;
//
//        // ���������������ڵ���200�Ĳ���������ִ�ж������
//        if (testCase.container_num <= 200) {
//            // ȷ����������ʱʹ��DROģʽ
//            pm.tm->model_type = ModelType::DRO;
//
//            // ���ݲ�������ִ����Ӧ�Ĳ���
//            if (testType.find("theta") != std::string::npos) {
//                vector<float> thetaValues = {
//                        0.0, 0.00001, 0.0001, 0.001, 0.01,
//                        0.1,
//                        0.15, 0.2, 0.25, 0.3
//                };
//                executeParameterTest(pm, bay, agv, "theta", thetaValues, pm.Theta);
//            }
//
//            if (testType.find("epsilon") != std::string::npos) {
//                vector<float> epsilonValues = {
//                        0.1, 0.2,
//                        0.3,
//                        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
//                };
//                executeParameterTest(pm, bay, agv, "epsilon", epsilonValues, pm.epsilon);
//            }
//
//            // �ָ�ԭʼģ������״̬
//            pm.tm->model_type = originalModelType;
//
//            if (pm.tm->warm_start){
//                vector<string> stragety = {
//                        "warm", "arc2", "warm_arc2", "arc2_arc3"
//                };
//                executeAccelerateTest(pm, bay, agv, "accelerate", stragety);
//            }
//        }

        // ����������������ִ�б�׼����
//        executeStandardOrRobustTest(pm);

        if (pm.isPrint) {
            std::cout << "****************************" << endl;
        }

//        truck_DRO.reset();



        //delete solution;

    }
}

void TestUnit::get_fileName() {

    stringstream ss;
    ss << "../../../result/" << testType << "_";

    switch (solve_method) {
        case GUROBI:
            ss << "GUROBI";
            break;
        case LBBD_GUROBI:
            ss << "LBBD_GUROBI";
            break;
        case LBBD_RULE:
            ss << "LBBD_RULE";
            break;
        case HEURISTICS:
            ss << "HEURISTICS";
            break;
    }
    ss << "_";

    // ����ģ�����������ļ���
    switch (model_type) {
        case ModelType::DETERMINISTIC:
            ss << "DETERMINISTIC";
            break;
        case ModelType::SAA:
            ss << "SAA";
            break;
        case ModelType::DRO:
            ss << "DRO";
            break;
        default:
            ss << "DETERMINISTIC";
            break;
    }

    ss << ".txt";
    fileName_printOut = ss.str();
}

void TestUnit::get_test_data(){
    if (testType.find("small") != std::string::npos){
        testCases.push_back(TestCase(20, 1, 1));
        testCases.push_back(TestCase(40, 1, 1));
        testCases.push_back(TestCase(60, 1, 1));
        testCases.push_back(TestCase(80, 1, 1));
        testCases.push_back(TestCase(100, 1, 1));
        testCases.push_back(TestCase(120, 2, 1));
        testCases.push_back(TestCase(140, 2, 1));
        testCases.push_back(TestCase(160, 2, 1));
        testCases.push_back(TestCase(180, 2, 1));
        testCases.push_back(TestCase(200, 2, 1));
        testCases.push_back(TestCase(220, 3, 1));
        testCases.push_back(TestCase(240, 3, 1));
        testCases.push_back(TestCase(260, 3, 1));
        testCases.push_back(TestCase(280, 3, 1));
        testCases.push_back(TestCase(300, 3, 1));
        testCases.push_back(TestCase(320, 4, 1));
        testCases.push_back(TestCase(340, 4, 1));
        testCases.push_back(TestCase(360, 4, 1));
        testCases.push_back(TestCase(380, 4, 1));
        testCases.push_back(TestCase(400, 4, 1));
        testCases.push_back(TestCase(420, 5, 1));
        testCases.push_back(TestCase(440, 5, 1));
        testCases.push_back(TestCase(460, 5, 1));
        testCases.push_back(TestCase(480, 5, 1));
        testCases.push_back(TestCase(500, 5, 1));
        testCases.push_back(TestCase(520, 6, 1));
        testCases.push_back(TestCase(540, 6, 1));
        testCases.push_back(TestCase(560, 6, 1));
        testCases.push_back(TestCase(580, 6, 1));
        testCases.push_back(TestCase(600, 6, 1));
        testCases.push_back(TestCase(620, 7, 1));
        testCases.push_back(TestCase(640, 7, 2));
        testCases.push_back(TestCase(660, 7, 2));
        testCases.push_back(TestCase(680, 7, 2));
        testCases.push_back(TestCase(700, 7, 2));
        testCases.push_back(TestCase(720, 8, 2));
        testCases.push_back(TestCase(740, 8, 2));
        testCases.push_back(TestCase(760, 8, 2));
        testCases.push_back(TestCase(780, 8, 2));
        testCases.push_back(TestCase(800, 8, 2));
        testCases.push_back(TestCase(820, 9, 2));
        testCases.push_back(TestCase(840, 9, 2));
        testCases.push_back(TestCase(860, 9, 2));
        testCases.push_back(TestCase(880, 9, 2));
        testCases.push_back(TestCase(900, 9, 2));
        testCases.push_back(TestCase(920, 10, 2));
        testCases.push_back(TestCase(940, 10, 2));
        testCases.push_back(TestCase(960, 10, 2));
        testCases.push_back(TestCase(980, 10, 2));
        testCases.push_back(TestCase(1000, 10, 2));
        testCases.push_back(TestCase(1020, 11, 2));
        testCases.push_back(TestCase(1040, 11, 2));
        testCases.push_back(TestCase(1060, 11, 2));
        testCases.push_back(TestCase(1080, 11, 2));
        testCases.push_back(TestCase(1100, 11, 2));
        testCases.push_back(TestCase(1120, 12, 2));
        testCases.push_back(TestCase(1140, 12, 2));
        testCases.push_back(TestCase(1160, 12, 2));
        testCases.push_back(TestCase(1180, 12, 2));
        testCases.push_back(TestCase(1200, 12, 2));
        testCases.push_back(TestCase(1220, 13, 2));
        testCases.push_back(TestCase(1240, 13, 2));
        testCases.push_back(TestCase(1260, 13, 3));
        testCases.push_back(TestCase(1280, 13, 3));
        testCases.push_back(TestCase(1300, 13, 3));
        testCases.push_back(TestCase(1320, 14, 3));
        testCases.push_back(TestCase(1340, 14, 3));
        testCases.push_back(TestCase(1360, 14, 3));
        testCases.push_back(TestCase(1380, 14, 3));
        testCases.push_back(TestCase(1400, 14, 3));
        testCases.push_back(TestCase(1420, 15, 3));
        testCases.push_back(TestCase(1440, 15, 3));
        testCases.push_back(TestCase(1460, 15, 3));
        testCases.push_back(TestCase(1480, 15, 3));
        testCases.push_back(TestCase(1500, 15, 3));
        testCases.push_back(TestCase(1520, 16, 3));
        testCases.push_back(TestCase(1540, 16, 3));
        testCases.push_back(TestCase(1560, 16, 3));
        testCases.push_back(TestCase(1580, 16, 3));
        testCases.push_back(TestCase(1600, 16, 3));
        testCases.push_back(TestCase(1620, 17, 3));
        testCases.push_back(TestCase(1640, 17, 3));
        testCases.push_back(TestCase(1660, 17, 3));
        testCases.push_back(TestCase(1680, 17, 3));
        testCases.push_back(TestCase(1700, 17, 3));
        testCases.push_back(TestCase(1720, 18, 3));
        testCases.push_back(TestCase(1740, 18, 3));
        testCases.push_back(TestCase(1760, 18, 3));
        testCases.push_back(TestCase(1780, 18, 3));
        testCases.push_back(TestCase(1800, 18, 3));
        testCases.push_back(TestCase(1820, 19, 3));
        testCases.push_back(TestCase(1840, 19, 3));
        testCases.push_back(TestCase(1860, 19, 3));
        testCases.push_back(TestCase(1880, 19, 3));
        testCases.push_back(TestCase(1900, 19, 3));
        testCases.push_back(TestCase(1920, 20, 4));
        testCases.push_back(TestCase(1940, 20, 4));
        testCases.push_back(TestCase(1960, 20, 4));
        testCases.push_back(TestCase(1980, 20, 4));
        testCases.push_back(TestCase(2000, 20, 4));
        testCases.push_back(TestCase(2020, 21, 4));
        testCases.push_back(TestCase(2040, 21, 4));
        testCases.push_back(TestCase(2060, 21, 4));
        testCases.push_back(TestCase(2080, 21, 4));
        testCases.push_back(TestCase(2100, 21, 4));
        testCases.push_back(TestCase(2120, 22, 4));
        testCases.push_back(TestCase(2140, 22, 4));
        testCases.push_back(TestCase(2160, 22, 4));
        testCases.push_back(TestCase(2180, 22, 4));
        testCases.push_back(TestCase(2200, 22, 4));
        testCases.push_back(TestCase(2220, 23, 4));
        testCases.push_back(TestCase(2240, 23, 4));
        testCases.push_back(TestCase(2260, 23, 4));
        testCases.push_back(TestCase(2280, 23, 4));
        testCases.push_back(TestCase(2300, 23, 4));
        testCases.push_back(TestCase(2320, 24, 4));
        testCases.push_back(TestCase(2340, 24, 4));
        testCases.push_back(TestCase(2360, 24, 4));
        testCases.push_back(TestCase(2380, 24, 4));
        testCases.push_back(TestCase(2400, 24, 4));
        testCases.push_back(TestCase(2420, 25, 4));
        testCases.push_back(TestCase(2440, 25, 4));
        testCases.push_back(TestCase(2460, 25, 4));
        testCases.push_back(TestCase(2480, 25, 4));
        testCases.push_back(TestCase(2500, 25, 4));
        testCases.push_back(TestCase(2520, 26, 4));
        testCases.push_back(TestCase(2540, 26, 5));
        testCases.push_back(TestCase(2560, 26, 5));
        testCases.push_back(TestCase(2580, 26, 5));
        testCases.push_back(TestCase(2600, 26, 5));
        testCases.push_back(TestCase(2620, 27, 5));
        testCases.push_back(TestCase(2640, 27, 5));
        testCases.push_back(TestCase(2660, 27, 5));
        testCases.push_back(TestCase(2680, 27, 5));
        testCases.push_back(TestCase(2700, 27, 5));
        testCases.push_back(TestCase(2720, 28, 5));
        testCases.push_back(TestCase(2740, 28, 5));
        testCases.push_back(TestCase(2760, 28, 5));
        testCases.push_back(TestCase(2780, 28, 5));
        testCases.push_back(TestCase(2800, 28, 5));
        testCases.push_back(TestCase(2820, 29, 5));
        testCases.push_back(TestCase(2840, 29, 5));
        testCases.push_back(TestCase(2860, 29, 5));
        testCases.push_back(TestCase(2880, 29, 5));
        testCases.push_back(TestCase(2900, 29, 5));
        testCases.push_back(TestCase(2920, 30, 5));
        testCases.push_back(TestCase(2940, 30, 5));
        testCases.push_back(TestCase(2960, 30, 5));
        testCases.push_back(TestCase(2980, 30, 5));
        testCases.push_back(TestCase(3000, 30, 5));
        testCases.push_back(TestCase(3020, 31, 5));
        testCases.push_back(TestCase(3040, 31, 5));
        testCases.push_back(TestCase(3060, 31, 5));
        testCases.push_back(TestCase(3080, 31, 5));
        testCases.push_back(TestCase(3100, 31, 5));
        testCases.push_back(TestCase(3120, 32, 5));
        testCases.push_back(TestCase(3140, 32, 5));
        testCases.push_back(TestCase(3160, 32, 5));
        testCases.push_back(TestCase(3180, 32, 6));
        testCases.push_back(TestCase(3200, 32, 6));
        testCases.push_back(TestCase(3220, 33, 6));
        testCases.push_back(TestCase(3240, 33, 6));
        testCases.push_back(TestCase(3260, 33, 6));
        testCases.push_back(TestCase(3280, 33, 6));
        testCases.push_back(TestCase(3300, 33, 6));
        testCases.push_back(TestCase(3320, 34, 6));
        testCases.push_back(TestCase(3340, 34, 6));
        testCases.push_back(TestCase(3360, 34, 6));
        testCases.push_back(TestCase(3380, 34, 6));
        testCases.push_back(TestCase(3400, 34, 6));
        testCases.push_back(TestCase(3420, 35, 6));
        testCases.push_back(TestCase(3440, 35, 6));
        testCases.push_back(TestCase(3460, 35, 6));
        testCases.push_back(TestCase(3480, 35, 6));
        testCases.push_back(TestCase(3500, 35, 6));
        testCases.push_back(TestCase(3520, 36, 6));
        testCases.push_back(TestCase(3540, 36, 6));
        testCases.push_back(TestCase(3560, 36, 6));
        testCases.push_back(TestCase(3580, 36, 6));
        testCases.push_back(TestCase(3600, 36, 6));
        testCases.push_back(TestCase(3620, 37, 6));
        testCases.push_back(TestCase(3640, 37, 6));
        testCases.push_back(TestCase(3660, 37, 6));
        testCases.push_back(TestCase(3680, 37, 6));
        testCases.push_back(TestCase(3700, 37, 6));
        testCases.push_back(TestCase(3720, 38, 6));
        testCases.push_back(TestCase(3740, 38, 6));
        testCases.push_back(TestCase(3760, 38, 6));
        testCases.push_back(TestCase(3780, 38, 6));
        testCases.push_back(TestCase(3800, 38, 6));
        testCases.push_back(TestCase(3820, 39, 7));
        testCases.push_back(TestCase(3840, 39, 7));
        testCases.push_back(TestCase(3860, 39, 7));
        testCases.push_back(TestCase(3880, 39, 7));
        testCases.push_back(TestCase(3900, 39, 7));
        testCases.push_back(TestCase(3920, 40, 7));
        testCases.push_back(TestCase(3940, 40, 7));
        testCases.push_back(TestCase(3960, 40, 7));
        testCases.push_back(TestCase(3980, 40, 7));
        testCases.push_back(TestCase(4000, 40, 7));
        testCases.push_back(TestCase(4020, 41, 7));
        testCases.push_back(TestCase(4040, 41, 7));
        testCases.push_back(TestCase(4060, 41, 7));
        testCases.push_back(TestCase(4080, 41, 7));
        testCases.push_back(TestCase(4100, 41, 7));
        testCases.push_back(TestCase(4120, 42, 7));
        testCases.push_back(TestCase(4140, 42, 7));
        testCases.push_back(TestCase(4160, 42, 7));
        testCases.push_back(TestCase(4180, 42, 7));
        testCases.push_back(TestCase(4200, 42, 7));
        testCases.push_back(TestCase(4220, 43, 7));
        testCases.push_back(TestCase(4240, 43, 7));
        testCases.push_back(TestCase(4260, 43, 7));
        testCases.push_back(TestCase(4280, 43, 7));
        testCases.push_back(TestCase(4300, 43, 7));
        testCases.push_back(TestCase(4320, 44, 7));
        testCases.push_back(TestCase(4340, 44, 7));
        testCases.push_back(TestCase(4360, 44, 7));
        testCases.push_back(TestCase(4380, 44, 7));
        testCases.push_back(TestCase(4400, 44, 7));
        testCases.push_back(TestCase(4420, 45, 8));
        testCases.push_back(TestCase(4440, 45, 8));
        testCases.push_back(TestCase(4460, 45, 8));
        testCases.push_back(TestCase(4480, 45, 8));
        testCases.push_back(TestCase(4500, 45, 8));
        testCases.push_back(TestCase(4520, 46, 8));
        testCases.push_back(TestCase(4540, 46, 8));
        testCases.push_back(TestCase(4560, 46, 8));
        testCases.push_back(TestCase(4580, 46, 8));
        testCases.push_back(TestCase(4600, 46, 8));
        testCases.push_back(TestCase(4620, 47, 8));
        testCases.push_back(TestCase(4640, 47, 8));
        testCases.push_back(TestCase(4660, 47, 8));
        testCases.push_back(TestCase(4680, 47, 8));
        testCases.push_back(TestCase(4700, 47, 8));
        testCases.push_back(TestCase(4720, 48, 8));
        testCases.push_back(TestCase(4740, 48, 8));
        testCases.push_back(TestCase(4760, 48, 8));
        testCases.push_back(TestCase(4780, 48, 8));
        testCases.push_back(TestCase(4800, 48, 8));
        testCases.push_back(TestCase(4820, 49, 8));
        testCases.push_back(TestCase(4840, 49, 8));
        testCases.push_back(TestCase(4860, 49, 8));
        testCases.push_back(TestCase(4880, 49, 8));
        testCases.push_back(TestCase(4900, 49, 8));
        testCases.push_back(TestCase(4920, 50, 8));
        testCases.push_back(TestCase(4940, 50, 8));
        testCases.push_back(TestCase(4960, 50, 8));
        testCases.push_back(TestCase(4980, 50, 8));
        testCases.push_back(TestCase(5000, 50, 8));
    }
    else if(testType.find("large") != std::string::npos){
        // �������ڽ������ܵ���չ
    }
    else if(testType.find("test") != std::string::npos){
        // ���������԰���

//        // ��һ�飺�������� 5-50��AGV 1-3�����վ 1-2��20��������
//
//        // ����1: 5-15����1-2 AGV��1���վ
        testCases.push_back(TestCase(5, 1, 1));      // ��С��ģ����
        testCases.push_back(TestCase(7, 1, 1));      // ��AGVС��ģ
        testCases.push_back(TestCase(10, 1, 1));     // ��׼С��ģ-��AGV
        testCases.push_back(TestCase(12, 2, 1));     // ���ɵ�˫AGV
        testCases.push_back(TestCase(15, 2, 1));     // ˫AGVС��ģ
//
//        // ����2: 18-28����2 AGV��1-2���վ
        testCases.push_back(TestCase(18, 2, 1));     // ˫AGV�е�����
        testCases.push_back(TestCase(20, 2, 1));     // ��׼��ģ-˫AGV
        testCases.push_back(TestCase(22, 2, 1));     // ˫AGV��������
        testCases.push_back(TestCase(25, 2, 2));     // ˫AGV˫���վ
        testCases.push_back(TestCase(28, 2, 2));     // ˫AGV˫���վ��������
//
//        // ����3: 30-40����2-3 AGV��2���վ
        testCases.push_back(TestCase(30, 2, 2));     // �еȹ�ģ-˫AGV
        testCases.push_back(TestCase(32, 3, 2));     // ���ɵ���AGV
        testCases.push_back(TestCase(35, 3, 2));     // ��AGV��׼��ģ
        testCases.push_back(TestCase(38, 3, 2));     // ��AGV��������
        testCases.push_back(TestCase(40, 3, 2));     // ��AGV�и�������
//
//        // ����4: 42-50����3 AGV��2���վ
        testCases.push_back(TestCase(42, 3, 2));     // ��AGV��������
        testCases.push_back(TestCase(45, 3, 2));     // ��AGV����������
        testCases.push_back(TestCase(47, 3, 2));     // ������ģ
        testCases.push_back(TestCase(49, 3, 2));     // �ӽ�����ģ
        testCases.push_back(TestCase(50, 3, 2));     // ��һ������ģ
//
//        // �ڶ��飺�������� 50-300��AGV 4-6�����վ 2-4��20��������
//
//        // ����1: 50-100����4 AGV��2���վ
        testCases.push_back(TestCase(50, 4, 2));     // ��С��ģ-��AGV
        testCases.push_back(TestCase(60, 4, 2));     // ��AGV�͸���
        testCases.push_back(TestCase(70, 4, 2));     // ��AGV�еȸ���
        testCases.push_back(TestCase(85, 4, 2));     // ��AGV�ϸ߸���
        testCases.push_back(TestCase(100, 4, 2));    // ��AGV�߸���
//
//        // ����2: 120-160����4-5 AGV��2-3���վ
        testCases.push_back(TestCase(120, 4, 3));    // ��AGV�����վ
        testCases.push_back(TestCase(130, 5, 3));    // ���ɵ���AGV
        testCases.push_back(TestCase(140, 5, 3));    // ��AGV�еȸ���
        testCases.push_back(TestCase(150, 5, 3));    // ��AGV�ϸ߸���
        testCases.push_back(TestCase(160, 5, 3));    // ��AGV�߸���

        // ����3: 180-240����5-6 AGV��3-4���վ
//        testCases.push_back(TestCase(180, 5, 3));    // ��AGV�߸���
        testCases.push_back(TestCase(200, 6, 4));    // ��AGV�ĳ��վ
        testCases.push_back(TestCase(210, 6, 4));    // ���ɵ���AGV
        testCases.push_back(TestCase(225, 6, 4));    // ��AGV�ϸ߸���
        testCases.push_back(TestCase(240, 6, 4));    // ��AGV�߸���

        // ����4: 250-300����6 AGV��4���վ
        testCases.push_back(TestCase(250, 6, 4));    // ��AGV�߸���
        testCases.push_back(TestCase(270, 6, 4));    // ��AGV���߸���
        testCases.push_back(TestCase(280, 6, 4));    // ��AGV�ӽ������
        testCases.push_back(TestCase(290, 6, 4));    // ��AGV������ģ
        testCases.push_back(TestCase(300, 6, 4));    // ����ģ����
    }
}


void TestUnit::print_begin(const Parameter &truck_pm) {
    //sharedGenerator.seed(1);
    //std::mt19937 verificationGenerator(1);
    // ʹ��ȫ������������������
    normal_distribution<double> randomDist(0.5, 0.1);
    double randomValue = randomDist(sharedGenerator);
    //std::cout << "Random seed verification number: " << randomValue << std::endl;
    if (truck_pm.isPrint) {
        std::cout << "****************************" << std::endl;

        // ��ӡ�������ȷ��ÿ������ʹ����ͬ���������
        std::cout << "Random seed verification number: " << randomValue << std::endl;
    }
}

TestUnit TestUnit::getCopy(){
    std::string testType = this->testType;
    bool Totxt = this->Totxt;
    SolveMethod solve_method = this->solve_method;
    bool accelerate = this->warm_start;
    // ����ʱʹ��DROģ������
    TestUnit testModel(testType, solve_method, this->warm_start, ModelType::DRO, Totxt);
    return testModel;
}

void TestUnit::executeParameterTest(Parameter &truck_pm, const string &paramName, vector<float> &testValues,
                                    double &defaultValue) {
    std::cout << "****************************" << std::endl;
    std::cout << "***************||" << paramName << "_test||***************" << std::endl;

    // ����ԭʼֵ
    double originalValue = defaultValue;

    // ��ÿ������ִֵ���Ż�
    for (auto value : testValues) {
        if (paramName == "theta") {
            truck_pm.Theta = value;
        } else if (paramName == "epsilon") {
            truck_pm.epsilon = value;
        }
        executeModelOptimization(truck_pm, ModelType::DRO);
    }

    // �ָ�ԭʼֵ
    defaultValue = originalValue;

}

void TestUnit::executeAccelerateTest(Parameter &agv_pm, const string &paramName, vector<string> stragetyes) {
    std::cout << "****************************" << std::endl;
    std::cout << "***************|| without" << paramName << "_test||***************" << std::endl;

//    string originalValue = pm.accelerate_stragety;
////    pm.max_time /= 2;
//    // ��ÿ������ִֵ���Ż�
//    for (auto stragety : stragetyes) {
//        pm.accelerate_stragety = stragety;
//        executeModelOptimization(pm, ModelType::DRO);
//    }
//    pm.accelerate_stragety = originalValue;
////    pm.max_time *= 2;
}

void TestUnit::executeModelOptimization(Parameter &pm, ModelType testModelType) {
    // ����ԭʼģ������
    ModelType originalModelType = pm.tm->model_type;

    // ���ò����õ�ģ������
    pm.tm->model_type = testModelType;

    try
    {
        // ����Ӧ����Ӿ�����Ż�����
        // ���磺
        // Model truck_Model(pm);
        // auto solution = truck_Model.solve(Model::EXACT);
    }
    catch (const GRBException& e) {
        std::cerr << "Gurobi exception: " << e.getMessage() << std::endl;
        std::cout << e.getMessage() << endl;
    }
    catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << std::endl;
        std::cout << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "An unknown exception occurred." << std::endl;
        std::cout << "An unknown exception occurred." << std::endl;
    }

    // �ָ�ԭʼģ������
    pm.tm->model_type = originalModelType;
}

void TestUnit::executeStandardOrRobustTest(Parameter &pm) {
    switch (pm.tm->model_type) {
        case ModelType::DETERMINISTIC:
            // ִ��ȷ���Բ��ԣ�Ҳ���Զ����������ģ�����ͽ��жԱ�
            executeModelOptimization(pm, ModelType::DETERMINISTIC);
            // ��ѡ��Ҳ����DROģ�ͽ��жԱ�
            executeModelOptimization(pm, ModelType::DRO);
            break;
        case ModelType::SAA:
            // ִ��SAA����
            executeModelOptimization(pm, ModelType::SAA);
            break;
        case ModelType::DRO:
            // �������ǲ�������ʱִ��һ��DRO�Ż�
            if (testType.find("theta") == std::string::npos &&
                testType.find("epsilon") == std::string::npos) {
                executeModelOptimization(pm, ModelType::DRO);
            }
            break;
        default:
            // Ĭ��ִ��ȷ���Բ���
            executeModelOptimization(pm, ModelType::DETERMINISTIC);
            break;
    }
}