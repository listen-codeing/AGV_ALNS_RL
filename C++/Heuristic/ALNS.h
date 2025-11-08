//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_ALNS_H
#define QC_AGV_ALNS_H

#include <time.h>
#include <set>
#include <string>
#include "Statistics.h"
#include "ALNS_Iteration_Status.h"


class ISolution;
class IAcceptanceModule;
class ALNS_Parameters;
class AOperatorManager;
class IBestSolutionManager;
class ILocalSearchManager;

class Statistics;
class ALNS_Parameters;
class TruckResult;
class ARepairOperator;

class ALNS {
private:
    ISolution* currentSolution;
    IAcceptanceModule* acceptanceCriterion;
    ALNS_Parameters* param;
    AOperatorManager* opManager;
    IBestSolutionManager* bestSolManager;
    ILocalSearchManager* lsManager;
    size_t nbIterationsWC; // 上一次权重更新之后的迭代次数
    size_t nbIterations; // 当前迭代次数
    size_t nbIterationsWithoutImprovement; // 当前没有提高的代数
    size_t nbIterationsWithoutImprovementCurrent; //从当前解开始没有提高的代数
    size_t nbIterationsWithoutTransition; // 当前没有转移的代数,由于存在模拟退火的机制，因此是否提高、是否转换是不一样的
    size_t nbIterationsWithoutLocalSearch; // 自从上次调用局部搜索之后的迭代次数
    clock_t startingTime;
    clock_t endTime;
    double elapsedTime;
    double lowerBound; // 当前最优解的下界
    std::set<std::size_t> knownKeys; // 一个包含哈希码的集合
    Statistics stats; // 用来统计求解过程的信息
    ALNS_Iteration_Status status; // 上一次迭代的对象
    std::string name; // 算法名称

public:
    ALNS(std::string instanceName,
         ISolution& initialSolution,
         IAcceptanceModule& acceptanceCrit,
         ALNS_Parameters& parameters,
         AOperatorManager& opMan,
         IBestSolutionManager& solMan,
         ILocalSearchManager& lsMan);
    virtual ~ALNS();
    bool solve();
    bool checkAgainstKnownSolution(ISolution& sol); // 检验这个解是否已经被找到过
    void performOneIteration(); // 执行一次迭代
    bool isStoppingCriterionMet(); // 检验是否满足停止准则
    bool isNewBest(ISolution* newSol); // 检验这个解是否是最优解
    size_t getNumberKnownSolutions(){return knownKeys.size();}; // 返回已知解的数量
    bool transitionCurrentSolution(ISolution* newSol); // 决定是否应该接受新解
    IBestSolutionManager* getBestSolutionManager(){return bestSolManager;};  // 返回最优解管理器的指针
    //void getResult(AGVResult& result);

    void end();

};


#endif //QC_AGV_ALNS_H
