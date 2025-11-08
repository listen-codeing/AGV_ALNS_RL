//
// Created by limin on 2023/9/7.
//


#include "ALNS_Parameters.h"
#include <iostream>
#include <sstream>
#include <fstream>
//#include "../Node.h"
#include "../AGV.h"
#include "../Parameter.h"
using namespace std;

ALNS_Parameters::ALNS_Parameters(Parameter* pm, const std::vector<std::shared_ptr<AGV>>& trucks)
    : pm(pm), trucks(trucks)
{
    // The variables are initialized with default values.
    maxNbIterations = 200;

    maxRunningTime = 3600;

    maxNbIterationsNoImp = 20;

    stopCrit = ALL;

    noise = false;

    timeSegmentsIt = 100;

    nbItBeforeReinit = 100;

    sigma1 = 33;

    sigma2 = 20;

    sigma3 = 15;

    similarity1 = 2;

    similarity2 = 4;

    rho = 0.1;

    minimumWeight = 0.1;

    maximumWeight = 5;

    acKind = SA; // 模拟退火方法

    acPath = "";

    logFrequency = 10;

    performLocalSearch = true; // 采用局部搜索

    probabilityOfNoise = 0;

    reloadFrequency = maxNbIterations/5;

    lock = false;
    //破坏比例：在破坏阶段（移除任务），任务移除比例的最小值（10%）和最大值（40%）。
    minDestroyPerc = 0.4;

    maxDestroyPerc = 0.9;

    maximumDestroy = maxDestroyPerc * trucks[0]->task_sequence.size();

    minimunDestroy = minDestroyPerc * trucks[0]->task_sequence.size();


    //agv->graph.calTasksSimilarity(this);


}

ALNS_Parameters::~ALNS_Parameters()
{

}
