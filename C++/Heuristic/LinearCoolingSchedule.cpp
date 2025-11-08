//
// Created by limin on 2023/9/7.
//

#include <assert.h>
#include "LinearCoolingSchedule.h"
#include "CoolingSchedule_Parameters.h"
#include "ISolution.h"

LinearCoolingSchedule::LinearCoolingSchedule(ISolution& initSol, CoolingSchedule_Parameters& csParam, size_t nbIterations) {
    currentTemperature = (csParam.setupPercentage* initSol.getObjectiveValue()) / (-log(0.5));
    amountRemove = currentTemperature/static_cast<double>(nbIterations);

}

LinearCoolingSchedule::LinearCoolingSchedule(double startingTemperature, size_t nbIterations) {
    assert(nbIterations>0);
    assert(startingTemperature>=0);
    currentTemperature = startingTemperature;
    amountRemove = startingTemperature/static_cast<double>(nbIterations);

}

LinearCoolingSchedule::~LinearCoolingSchedule() {
    // Nothing to be done.
}

double LinearCoolingSchedule::getCurrentTemperature()
{
    currentTemperature-= amountRemove;
    if(currentTemperature < 0)
    {
        currentTemperature = 0;
    }
    assert(currentTemperature>=0);
    return currentTemperature;
}