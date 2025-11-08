//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_AOPERATORMANAGER_H
#define QC_AGV_AOPERATORMANAGER_H

class ALocalSearchOperator;
class ANeighSearchOperator;
class ALNS_Iteration_Status;
class Statistics;
class AOperator;
class ARepairOperator;
class ADestroyOperator;

class AOperatorManager
{
public:
    virtual void recomputeWeights()=0;
    // 更新得分
    virtual void updateScores(ADestroyOperator& des, ARepairOperator& rep, ALNS_Iteration_Status& status)=0;
    // 优化开始
    virtual void startSignal()=0;
    //! Destroy the operators registered to this operator manager. 销毁注册的算子
    virtual void end()=0;
    //! Simple setter.
    void setStatistics(Statistics* statistics){stats = statistics;};
    virtual ARepairOperator& selectRepairOperator()=0;
    virtual ADestroyOperator& selectDestroyOperator()=0;

protected:
    //! A pointer to the instance of the statistics module.
    Statistics* stats;
};

#endif //QC_AGV_AOPERATORMANAGER_H
