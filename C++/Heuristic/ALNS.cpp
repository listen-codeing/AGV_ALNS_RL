//
// Created by limin on 2023/9/7.
//

#include "ALNS.h"
#include "ALNS_Parameters.h"
#include "ISolution.h"
#include "OperatorManager.h"
#include "IBestSolutionManager.h"
#include "IAcceptanceModule.h"
#include "ILocalSearchManager.h"
#include "AGV_solution.h"
//#include "../Result.h"
#include "ARepairOperator.h"
#include "ADestroyOperator.h"

using namespace std;

ALNS::ALNS(string instanceName,
           ISolution& initialSolution,
           IAcceptanceModule& acceptanceCrit,
           ALNS_Parameters& parameters,
           AOperatorManager& opMan,
           IBestSolutionManager& bestSolMan,
           ILocalSearchManager& lsMan)
{
    name = instanceName;
    currentSolution = initialSolution.getCopy();
    acceptanceCriterion = &acceptanceCrit;
    param = &parameters;
    lowerBound = -DBL_MAX;
    nbIterationsWC = 0;
    nbIterations = 0;
    nbIterationsWithoutImprovement = 0;
    opManager = &opMan;
    bestSolManager = &bestSolMan;
    lsManager = &lsMan;
    opManager->setStatistics(&stats);
    // We add the initial solution in the best solution manager.
    bestSolManager->isNewBestSolution(initialSolution);
    nbIterationsWithoutImprovementCurrent = 0;
    nbIterationsWithoutTransition = 0;
    nbIterationsWithoutLocalSearch = 0;
    srand(1);
}

ALNS::~ALNS()
{
    delete currentSolution;
}

bool ALNS::solve()
{

    this->param->setMaxNbIterations(50000);
    this->param->setMaxNbIterationsNoImp(50000);
    this->param->setReloadFrequency(this->param->getMaxNbIterations()/3);
    //this->param->setMaxNbIterationsNoImp(this->param->getMaxNbIterations()/2);
    //this->param->setLogFrequency(this->param->getMaxNbIterations()/4);
    this->param->setLogFrequency(1000);
    startingTime = clock(); // ��¼��ʼʱ��
    param->setLock(); // �趨��
    acceptanceCriterion->startSignal();//����׼��ĳ�ʼ״̬
    opManager->startSignal();// �������ӹ������ĳ�ʼ״̬
    stats.setStart(); // ״̬�ĳ�ʼ״̬

    while(!isStoppingCriterionMet())
    {

        performOneIteration();
    }

    // ��ӡ��������Ƿ����
    string pathGlob = param->getStatsGlobPath();
    pathGlob += name;
    pathGlob += ".txt";
    string pathOp = param->getStatsOpPath();
    pathOp += name;
    pathOp += ".txt";
    stats.generateStatsFile(pathGlob,pathOp);
    endTime = clock();
    double elapsed = static_cast<double>(endTime - startingTime) / CLOCKS_PER_SEC;

    cout << "�����ʱ: " << elapsed <<" ";
    return (*(bestSolManager->begin()))->isFeasible();
}

void ALNS::performOneIteration(){
    // ��ʼ��״̬��

    status.partialReinit();

    ADestroyOperator& destroy = opManager->selectDestroyOperator();
    ARepairOperator& repair = opManager->selectRepairOperator();
    ISolution* newSolution = currentSolution->getDeepCopy();

    // ÿ�����ٴ����һ�������Ϣ
    if(nbIterations  % param->getLogFrequency() == 0 || nbIterations == param->getMaxNbIterations() - 1)
    {

        //cout << "[ALNS] it. " << nbIterations << " best sol: " << (*(bestSolManager->begin()))->getObjectiveValue() << " get best solutions: " << bestSolManager->getBestSols().size() << endl;
        //cout << "[ALNS] it. " << nbIterations << "   best sol: " << (*(bestSolManager->begin()))->getObjectiveValue() << "   nb known solutions: " << knownKeys.size() << endl;

    }

    destroy.destroySolution(*newSolution);
    // ����״̬
    status.setAlreadyDestroyed(ALNS_Iteration_Status::TRUE);
    status.setAlreadyRepaired(ALNS_Iteration_Status::FALSE);

    repair.repairSolution(*newSolution);
    status.setAlreadyRepaired(ALNS_Iteration_Status::TRUE);

    //���µ�������
    nbIterations++;
    status.setIterationId(nbIterations);
    nbIterationsWC++;

    double newCost = newSolution->getObjectiveValue();
    isNewBest(newSolution); // �ж��������Ľ��ǲ����µ����Ž�
    checkAgainstKnownSolution(*newSolution); // �ж������ɵĽ�֮ǰ��û�г��ֹ�
    bool betterThanCurrent = (*newSolution)<(*currentSolution); //�ж������ɵĽ�͵�ǰ��˭����
    if(betterThanCurrent)
    {
        nbIterationsWithoutImprovementCurrent = 0;
        status.setImproveCurrentSolution(ALNS_Iteration_Status::TRUE);
    }
    else
    {
        nbIterationsWithoutImprovementCurrent++;
        status.setImproveCurrentSolution(ALNS_Iteration_Status::FALSE); // ���ε�����û�����
    }
    // ����״̬
    status.setNbIterationWithoutImprovementCurrent(nbIterationsWithoutImprovementCurrent);

    //param->getPerformLocalSearch()������Ҫ��Ҫ�þֲ�������lsManager->useLocalSearch(*newSolution,status)���ֲ�����֮����û�и���
    if(param->getPerformLocalSearch() && lsManager->useLocalSearch(*newSolution,status))
    {
        bestSolManager->isNewBestSolution(*newSolution);
    }
    //�ж��Ƿ���ܵ�ǰ�Ľ⡣
    bool transitionAccepted = transitionCurrentSolution(newSolution);
    // ������ܵ�ǰ��
    if(transitionAccepted)
    {
        status.setAcceptedAsCurrentSolution(ALNS_Iteration_Status::TRUE);
        nbIterationsWithoutTransition = 0;
    }
    else
    {
        status.setAcceptedAsCurrentSolution(ALNS_Iteration_Status::FALSE);
        nbIterationsWithoutTransition++;
    }
    status.setNbIterationWithoutTransition(nbIterationsWithoutTransition);
    //�ٸ�һ��LocalSearch������
    if(param->getPerformLocalSearch() && lsManager->useLocalSearch(*newSolution,status))
    {
        bestSolManager->isNewBestSolution(*newSolution);
        if(status.getAcceptedAsCurrentSolution() == ALNS_Iteration_Status::TRUE)
        {
            transitionCurrentSolution(newSolution);
        }
    }
    //��destroy,repair������Ȩ�ؽ��и���
    opManager->updateScores(destroy,repair,status);

    // ��¼ÿ�ε��������̵Ĳ���
    stats.addEntry(static_cast<double>(clock()-startingTime)/CLOCKS_PER_SEC,nbIterations,destroy.getName(),repair.getName(),newCost,currentSolution->getObjectiveValue(),(*(bestSolManager->begin()))->getObjectiveValue(),knownKeys.size());

    //����destroy,repair������Ȩ�ء����ڽ�����һ�����������Ժ�Ÿ��µģ����������param->getTimeSegmentsIt()��á�
    if(nbIterationsWC % param->getTimeSegmentsIt() == 0)
    {
        opManager->recomputeWeights();
        nbIterationsWC = 0;
    }

    //�������Ҫ������ǰ��ת������Ž��ٽ�����һ�ε���������
    currentSolution = bestSolManager->reloadBestSolution(currentSolution,status);

    delete newSolution;
}

// ����hash����ý��Ƿ���֮ǰ���ֹ��Ľ�
bool ALNS::checkAgainstKnownSolution(ISolution& sol)
{
    bool notKnownSolution = false;
    size_t keySol = sol.getHash();

    if(knownKeys.find(keySol) == knownKeys.end())
    {
        notKnownSolution = true;
        knownKeys.insert(keySol);
    }

    if(!notKnownSolution)
    {
        status.setAlreadyKnownSolution(ALNS_Iteration_Status::TRUE);
    }
    else
    {
        status.setAlreadyKnownSolution(ALNS_Iteration_Status::FALSE);
    }

    return notKnownSolution;
}


// �ж��Ƿ�Ϊ�µ����Ž⣬��������Ӧ����
bool ALNS::isNewBest(ISolution* newSol)
{
    if(bestSolManager->isNewBestSolution(*newSol))
    {
        status.setNewBestSolution(ALNS_Iteration_Status::TRUE);
        nbIterationsWithoutImprovement = 0;
        status.setNbIterationWithoutImprovement(nbIterationsWithoutImprovement);
        status.setNbIterationWithoutImprovementSinceLastReload(0);
        return true;
    }
    else
    {
        status.setNewBestSolution(ALNS_Iteration_Status::FALSE);
        nbIterationsWithoutImprovement++;
        status.setNbIterationWithoutImprovement(nbIterationsWithoutImprovement);
        status.setNbIterationWithoutImprovementSinceLastReload(status.getNbIterationWithoutImprovementSinceLastReload()+1);
        return false;
    }
}

// ����׼���ж��Ƿ���ܵ�ǰ��Ϊ�µĽ�
bool ALNS::transitionCurrentSolution(ISolution* newSol)
{
    if(acceptanceCriterion->transitionAccepted(*bestSolManager,*currentSolution,*newSol,status))
    {
//        nbIterationsWithoutImprovementCurrent = 0;
        status.setImproveCurrentSolution(ALNS_Iteration_Status::TRUE);
        delete currentSolution;
        currentSolution = newSol->getCopy();
        return true;
    }
    else
    {
        return false;
    }
}

bool ALNS::isStoppingCriterionMet() {


    // ���ָ���Ƿ�Ϊ��
    if (!bestSolManager || !param) {
        cerr << "Error: bestSolManager or param is null!" << endl;
        return false;
    }

    // ��� bestSolManager �Ƿ��н�
    if (bestSolManager->begin() == bestSolManager->end()) {
        cerr << "Error: No solutions in bestSolManager!" << endl;
        return false;
    }

    // �ҵ������ſ��н�, ������Ŀ�꺯��ֵ�����½�
    if ((*(bestSolManager->begin()))->isFeasible() && (*(bestSolManager->begin()))->getObjectiveValue() == lowerBound) {

        return true;
    } else {

        switch (param->getStopCrit()) {
            case ALNS_Parameters::MAX_IT: {
                return nbIterations >= param->getMaxNbIterations();
            }
            case ALNS_Parameters::MAX_RT: {
                clock_t currentTime = clock();
                double elapsed = (static_cast<double>(currentTime - startingTime)) / CLOCKS_PER_SEC;
            }
            case ALNS_Parameters::MAX_IT_NO_IMP: {
                return nbIterationsWithoutImprovement >= param->getMaxNbIterationsNoImp();
            }
            case ALNS_Parameters::ALL: {
                // 1. 检查最大迭代次数
                if (nbIterations >= param->getMaxNbIterations()) {
                    return true;
                }
                // 2. 检查最大无改进迭代次数
                if (nbIterationsWithoutImprovement >= param->getMaxNbIterationsNoImp()) {
                    return true;
                }
                // 3. 检查最大运行时间
                clock_t currentTime = clock();
                double elapsed = (static_cast<double>(currentTime - startingTime)) / CLOCKS_PER_SEC;
                if (elapsed >= param->getMaxRunningTime()) {
                    return true;
                }

                return false; // 如果 ALL 模式下所有条件都不满足，则返回 false (不停止)
            }

            default: { // <--- 这样 default 才是用于捕获未处理的枚举值
                assert(false); // 仅在 param->getStopCrit() 返回一个未知的枚举值时触发
                return false;
            }
        }
    }
}
//bool ALNS::isStoppingCriterionMet()
//{
//    cout<<"here5.1-1"<<endl;
//    // �ҵ������ſ��н�,������Ŀ�꺯��ֵ�����½�
//    if((*(bestSolManager->begin()))->isFeasible() && (*(bestSolManager->begin()))->getObjectiveValue() == lowerBound)
//    {
//        cout<<"here5.1-2"<<endl;
//        return true;
//    }
//    else
//    {
//        cout<<"here5.1-3"<<endl;
//        switch(param->getStopCrit())
//        {
//            // �ﵽ����������
//            case ALNS_Parameters::MAX_IT: {
//                return nbIterations >= param->getMaxNbIterations();
//            }
//            // �������ʱ��
//            case ALNS_Parameters::MAX_RT: {
//                clock_t currentTime = clock();
//                double elapsed = (static_cast<double>(currentTime - startingTime)) / CLOCKS_PER_SEC;
//                return elapsed >= param->getMaxRunningTime();
//            }
//            // ��������û�Ľ��ﵽ���ֵ
//            case ALNS_Parameters::MAX_IT_NO_IMP: {
//                return nbIterationsWithoutImprovement >= param->getMaxNbIterationsNoImp();
//            }
//            case ALNS_Parameters::ALL: {
//                // �ﵽ����������
//                if(nbIterations >= param->getMaxNbIterations())
//                {
//                    return true;
//                }
//                // ��������û�Ľ��ﵽ���ֵ
//                if(nbIterationsWithoutImprovement >= param->getMaxNbIterationsNoImp())
//                {
//                    return true;
//                }
//                // �������ʱ��
//                clock_t currentTime = clock();
//                double elapsed = (static_cast<double>(currentTime - startingTime)) / CLOCKS_PER_SEC;
//                if(elapsed >= param->getMaxRunningTime())
//                {
//                    return true;
//                }
//                return false;
//            }
//
//            default: {
//                assert(false);
//                return false;
//            }
//        }
//    }
//}

// void ALNS::getResult(AGVResult& result)
// {
//     AGV_solution* agvSol = dynamic_cast<AGV_solution*>(bestSolManager->getBestSol());
//
//     result.obj = agvSol->getTotalObj();
//     result.agv_paths = agvSol->getRoutes();
//     result.agvNum = agvSol->getPathsSize();
//
// }

void ALNS::end()
{
    opManager->end();
    delete opManager;
    delete acceptanceCriterion;
    delete lsManager;
    delete bestSolManager;
}