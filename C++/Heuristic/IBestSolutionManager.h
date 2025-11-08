//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_IBESTSOLUTIONMANAGER_H
#define QC_AGV_IBESTSOLUTIONMANAGER_H

#include <list>

class ISolution;
class ALNS_Iteration_Status;

class IBestSolutionManager
{
public:
    // 判断是否是一个新的最优解
    virtual bool isNewBestSolution(ISolution& sol)=0;
    // 返回当前最好解的指针
    virtual std::list<ISolution*>::iterator begin()=0;
    virtual std::list<ISolution*>::iterator end()=0;
    virtual std::list<ISolution*>& getBestSols()= 0;
    //!可以将最优解赋值给当前解，并返回一个指向当前解的指针
    virtual ISolution* reloadBestSolution(ISolution* currSol, ALNS_Iteration_Status& status)=0;
    virtual ISolution* getBestSol()=0;
};

#endif //QC_AGV_IBESTSOLUTIONMANAGER_H
