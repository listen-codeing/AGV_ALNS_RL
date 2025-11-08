//
// Created by limin on 2023/9/7.
//

#include "SimulatedAnnealing.h"
#include "ICoolingSchedule.h"

SimulatedAnnealing::SimulatedAnnealing(ICoolingSchedule& cs) {
    coolingSchedule = &cs;
}

SimulatedAnnealing::~SimulatedAnnealing() {
//    delete coolingSchedule;
}

bool SimulatedAnnealing::transitionAccepted(IBestSolutionManager& bestSolutionManager,
                                            ISolution& currentSolution,
                                            ISolution& newSolution,
                                            ALNS_Iteration_Status& status)
{
    double temperature = coolingSchedule->getCurrentTemperature();
    if(newSolution < currentSolution)
    {
        return true;
    }
    else
    {
        double difference = newSolution.getObjectiveValue() - currentSolution.getObjectiveValue();
        double randomVal = static_cast<double>(rand())/static_cast<double>(RAND_MAX);
        return (exp(-1*difference/temperature)>randomVal);
    }
}

void SimulatedAnnealing::startSignal()
{
    coolingSchedule->startSignal();
}
