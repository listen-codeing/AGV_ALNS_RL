//
// Created by limin on 2023/10/28.
//

#ifndef QC_AGV_TESTMODEL_H
#define QC_AGV_TESTMODEL_H

#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include "Parameter.h"


class TestCase {
public:
    int truck_num;
    int charge_station_num;
    int container_num;

    double time_makespan;
    int taskNum = 10;

    TestCase(int container_num, int truck_num, int charge_station_num);

    void print_begin();

};

class TestUnit {
public:
    std::string testType;
    int omega_num = 200;
    std::vector<TestCase> testCases;
    bool Totxt;
    SolveMethod solve_method;
    bool warm_start;
    ModelType model_type;  // 已更新为ModelType枚举类型

    std::string fileName_printOut;
    std::string fileName_distance = "..\\..\\distance_matrix.csv";

    std::string fileName_SOC = "..\\..\\AGV充电时间预测_所有区间_随机森林.csv";

    // 构造函数 - 参数已更新为ModelType
    TestUnit(std::string testType, SolveMethod solve_method, bool warm_start, ModelType model_type, bool Totxt);

    void get_fileName();
    void run();
    void get_test_data();

    void print_begin(const Parameter &truck_pm);

    TestUnit getCopy();

private:
    // 用于执行不同类型的测试
    void executeParameterTest(Parameter &truck_pm, const string &paramName, vector<float> &testValues,
                              double &defaultValue);

    // 执行单次模型优化 - 参数已更新为ModelType
    void executeModelOptimization(Parameter &pm, ModelType testModelType);

    // 执行标准测试或鲁棒性测试
    void executeStandardOrRobustTest(Parameter &pm);

    // 设置输出重定向
    class OutputRedirector {
    public:
        OutputRedirector(bool toFile, const string& fileName) {
            if (toFile) {
                outFile.open(fileName);
                coutbuf = std::cout.rdbuf();
                std::cout.rdbuf(outFile.rdbuf());
            }
        }
        ~OutputRedirector() {
            if (coutbuf) {
                std::cout.rdbuf(coutbuf);
                outFile.close();
            }
        }
    private:
        std::ofstream outFile;
        std::streambuf* coutbuf = nullptr;
    };

    void executeAccelerateTest(Parameter &agv_pm, const string &paramName, vector<string> stragety);

};


#endif //QC_AGV_TESTMODEL_H