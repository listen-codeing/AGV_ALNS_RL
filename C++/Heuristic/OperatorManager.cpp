//
// Created by limin on 2023/9/7.
//

#include "OperatorManager.h"
#include "AOperator.h"

#include "ALNS_Parameters.h"
#include "Statistics.h"
// #include "ANeighSearchOperator.h"
// #include "ALocalSearchOperator.h"
#include "ALNS_Iteration_Status.h"
#include "ARepairOperator.h"
#include "ADestroyOperator.h"

OperatorManager::OperatorManager(ALNS_Parameters& param) {
    parameters = &param;
    sumWeightsRepair = 0;
    sumWeightsDestroy = 0;
    noise = false;

    performanceRepairOperatorsWithNoise = 1;
    performanceRepairOperatorsWithoutNoise = 1;
}

OperatorManager::~OperatorManager() {
    // Nothing to be done.
}


void OperatorManager::recomputeWeight(AOperator& op, double& sumW)
{
    double prevWeight = op.getWeight();
    sumW -= prevWeight;
    double currentScore = op.getScore();
    size_t nbCalls = op.getNumberOfCallsSinceLastEvaluation();
    double newWeight = (1-parameters->getRho())*prevWeight + parameters->getRho()*(static_cast<double>(nbCalls)/static_cast<double>(parameters->getTimeSegmentsIt()))*currentScore;
    // We ensure that the weight is within the bounds.
    if(newWeight > parameters->getMaximumWeight())
    {
        newWeight = parameters->getMaximumWeight();
    }
    if(newWeight < parameters->getMinimumWeight())
    {
        newWeight = parameters->getMinimumWeight();
    }

    sumW += newWeight;
    op.setWeight(newWeight);
    op.resetScore();
    op.resetNumberOfCalls();
}

void OperatorManager::recomputeWeights()
{
    // Retrieval of the number of calls for the statistics module.
    std::vector<size_t>* nbCalls = new std::vector<size_t>();
    for(size_t i = 0; i < repairOperators.size(); i++)
    {
        nbCalls->push_back(repairOperators[i]->getNumberOfCallsSinceLastEvaluation());
    }
    for(size_t i = 0; i < destroyOperators.size(); i++)
    {
        nbCalls->push_back(destroyOperators[i]->getNumberOfCallsSinceLastEvaluation());
    }


    // Weight recomputation for repair operators.
    for(size_t i = 0; i < repairOperators.size(); i++)
    {
        recomputeWeight(dynamic_cast<AOperator&>(*(repairOperators[i])),sumWeightsRepair);
    }

    //! Weight recomputation for destroy operators.
    for(size_t i = 0; i < destroyOperators.size(); i++)
    {
        recomputeWeight(dynamic_cast<AOperator&>(*(destroyOperators[i])),sumWeightsDestroy);
    }

    // Retrieval of the weights for the statistics module.
    std::vector<double>* weightsStats = new std::vector<double>();
    for(size_t i = 0; i < repairOperators.size(); i++)
    {
        weightsStats->push_back(repairOperators[i]->getWeight());
    }
    for(size_t i = 0; i < destroyOperators.size(); i++)
    {
        weightsStats->push_back(destroyOperators[i]->getWeight());
    }

    stats->addOperatorEntry(weightsStats,nbCalls);

    performanceRepairOperatorsWithNoise = 1;
    performanceRepairOperatorsWithoutNoise = 1;
}



AOperator& OperatorManager::selectOperator(std::vector<AOperator*>& vecOp, double sumW)
{
    double randomVal = static_cast<double>(rand())/static_cast<double>(RAND_MAX);
    double randomWeightPos = randomVal*sumW;
    double cumulSum = 0;
    //std::cout<<"randomWeightPos:"<<randomWeightPos<<"vecOp.size()"<<vecOp.size()<<std::endl;
    for(size_t i = 0; i < vecOp.size(); i++)
    {
        cumulSum += vecOp[i]->getWeight();
        if(cumulSum >= randomWeightPos)
        {
            if(noise)
            {
                vecOp[i]->setNoise();
            }
            else
            {
                vecOp[i]->unsetNoise();
            }
            vecOp[i]->increaseNumberOfCalls();
            return *(vecOp[i]);
        }
    }
    //assert(false);
    return *(vecOp.back());
}

void OperatorManager::updateScores(ADestroyOperator& des, ARepairOperator& rep, ALNS_Iteration_Status& status)
{
    // ������������Ž�
    if(status.getNewBestSolution() == ALNS_Iteration_Status::TRUE)
    {
        rep.setScore(rep.getScore()+parameters->getSigma1());
        des.setScore(des.getScore()+parameters->getSigma1());
        performanceRepairOperatorsWithNoise += 1;
        performanceRepairOperatorsWithoutNoise += 1;
    }
    // �����ǰ�ⱻ�Ľ�
    if(status.getImproveCurrentSolution() == ALNS_Iteration_Status::TRUE)
    {
        rep.setScore(rep.getScore()+parameters->getSigma2());
        des.setScore(des.getScore()+parameters->getSigma2());
        performanceRepairOperatorsWithNoise += 1;
        performanceRepairOperatorsWithoutNoise += 1;
    }
    // �����ǰ��û�Ľ� && ��ǰ��֮ǰû���ֹ� && ��ǰ�ⱻ������Ϊ�½�
    if(status.getImproveCurrentSolution() == ALNS_Iteration_Status::FALSE
       && status.getAcceptedAsCurrentSolution() == ALNS_Iteration_Status::TRUE
       && status.getAlreadyKnownSolution() == ALNS_Iteration_Status::FALSE)
    {
        rep.setScore(rep.getScore()+parameters->getSigma3());
        des.setScore(des.getScore()+parameters->getSigma3());
        performanceRepairOperatorsWithNoise += 1;
        performanceRepairOperatorsWithoutNoise += 1;
    }

    /* OLD VERSION */
    /*
    if(parameters->getNoise())
    {
        double randNoise = static_cast<double>(rand())/RAND_MAX;
        noise = (randNoise<parameters->getProbabilityOfNoise());
    }
    */

    /* NEW VERSION */

    if(parameters->getNoise())
    {
        double performanceRepairOperatorsGlobal = 0;
        performanceRepairOperatorsGlobal += performanceRepairOperatorsWithNoise;
        performanceRepairOperatorsGlobal += performanceRepairOperatorsWithoutNoise;

        double randomVal = static_cast<double>(rand())/RAND_MAX;
        double randomWeightPos = randomVal*performanceRepairOperatorsGlobal;
        noise = (randomWeightPos < performanceRepairOperatorsGlobal);
    }

}


void OperatorManager::end()
{
    for(int i = 0; i < repairOperators.size(); ++i)
    {
        delete repairOperators[i];
    }
    for(int i = 0; i < destroyOperators.size(); ++i)
    {
        delete destroyOperators[i];
    }
}

ARepairOperator& OperatorManager::selectRepairOperator()
{
    return dynamic_cast<ARepairOperator&>(selectOperator(repairOperators,sumWeightsRepair));
}

ADestroyOperator& OperatorManager::selectDestroyOperator()
{
    return dynamic_cast<ADestroyOperator&>(selectOperator(destroyOperators,sumWeightsDestroy));
}

void OperatorManager::addDestroyOperator(ADestroyOperator& destroyOperator)
{
    //TODO find out why the set.find()==set.end() does not work
    bool ok = true;
    for(size_t i = 0; i < parameters->getForbidenOperators().size() && ok; i++)
    {
        if(parameters->getForbidenOperators()[i] == destroyOperator.getName())
        {
            std::cout << destroyOperator.getName().c_str() << " forbidden" << std::endl;
            ok = false;
        }
    }
    if(ok)
    {
        destroyOperators.push_back(&destroyOperator);
        sumWeightsDestroy += destroyOperator.getWeight();
    }
}

void OperatorManager::addRepairOperator(ARepairOperator& repairOperator)
{
    //TODO find out why the set.find()==set.end() does not work

    bool ok = true;
    for(size_t i = 0; i < parameters->getForbidenOperators().size() && ok; i++)
    {
        if(parameters->getForbidenOperators()[i] == repairOperator.getName())
        {
            std::cout << repairOperator.getName().c_str() << " forbidden" << std::endl;
            ok = false;
        }
    }
    if(ok)
    {
        repairOperators.push_back(&repairOperator);
        sumWeightsRepair += repairOperator.getWeight();
    }
}

void OperatorManager::startSignal()
{
    std::vector<std::string>* names = new std::vector<std::string>();
    for(size_t i = 0; i < repairOperators.size(); i++)
    {
        names->push_back(repairOperators[i]->getName());
    }
    for(size_t i = 0; i < destroyOperators.size(); i++)
    {
        names->push_back(destroyOperators[i]->getName());
    }
    stats->addOperatorsNames(names);
}