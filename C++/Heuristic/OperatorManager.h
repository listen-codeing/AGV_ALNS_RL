//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_OPERATORMANAGER_H
#define QC_AGV_OPERATORMANAGER_H

#include <vector>
#include <string>
#include "AOperatorManager.h"

class AOperator;
class ALNS_Parameters;
class ALNS_Iteration_Status;

class OperatorManager: public AOperatorManager {
private:
    std::vector<AOperator*> repairOperators;
    std::vector<AOperator*> destroyOperators;
    double sumWeightsRepair;
    double sumWeightsDestroy;
    ALNS_Parameters* parameters;
    bool noise;
    double performanceRepairOperatorsWithNoise;
    double performanceRepairOperatorsWithoutNoise;

    AOperator& selectOperator(std::vector<AOperator*>& vecOp, double sumW);//根据权重选择操作符
    ARepairOperator& selectRepairOperator();
    ADestroyOperator& selectDestroyOperator();
    void recomputeWeight(AOperator& op, double& sumW);

public:
    OperatorManager(ALNS_Parameters& param);
    virtual ~OperatorManager();
    void recomputeWeights(); //重新计算每个算子的权重
    virtual void updateScores(ADestroyOperator& des, ARepairOperator& rep, ALNS_Iteration_Status& status);
    virtual void startSignal();
    void end();
    void addDestroyOperator(ADestroyOperator &destroyOperator);

    void addRepairOperator(ARepairOperator &repairOperator);
};


#endif //QC_AGV_OPERATORMANAGER_H
