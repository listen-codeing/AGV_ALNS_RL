//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_SIMULATEDANNEALING_H
#define QC_AGV_SIMULATEDANNEALING_H

#include "IAcceptanceModule.h"

class ISolution;
class ICoolingSchedule;
class ALNS_Iteration_Status;
class IBestSolutionManager;

class SimulatedAnnealing : public IAcceptanceModule {
private:
    ICoolingSchedule* coolingSchedule;
public:
    SimulatedAnnealing(ICoolingSchedule& cs);
    virtual ~SimulatedAnnealing();
    bool transitionAccepted(IBestSolutionManager& bestSolutionManager, ISolution& currentSolution, ISolution& newSolution, ALNS_Iteration_Status& status);
    virtual void startSignal();

};


#endif //QC_AGV_SIMULATEDANNEALING_H
