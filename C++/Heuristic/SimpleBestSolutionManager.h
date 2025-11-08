//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_SIMPLEBESTSOLUTIONMANAGER_H
#define QC_AGV_SIMPLEBESTSOLUTIONMANAGER_H

#include "IBestSolutionManager.h"
#include <list>

class ALNS_Parameters;

class SimpleBestSolutionManager : public IBestSolutionManager {
public:
    SimpleBestSolutionManager(ALNS_Parameters& param);
    virtual ~SimpleBestSolutionManager();
    bool isNewBestSolution(ISolution& sol);
    std::list<ISolution*>::iterator begin(){return bestSols.begin();};
    std::list<ISolution*>::iterator end(){return bestSols.end();};
    ISolution* reloadBestSolution(ISolution* currSol, ALNS_Iteration_Status& status);
    std::list<ISolution*>& getBestSols(){return bestSols;};
    ISolution* getBestSol();

private:
    std::list<ISolution*> bestSols;
    ALNS_Parameters* parameters;

};


#endif //QC_AGV_SIMPLEBESTSOLUTIONMANAGER_H
