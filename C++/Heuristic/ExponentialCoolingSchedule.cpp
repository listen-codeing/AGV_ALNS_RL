//
// Created by limin on 2023/9/14.
//

#include "ExponentialCoolingSchedule.h"
#include <math.h>
#include "ISolution.h"
#include "CoolingSchedule_Parameters.h"

ExponentialCoolingSchedule::ExponentialCoolingSchedule(ISolution& initSol, CoolingSchedule_Parameters& csParam) {
    // The fields are instantiated to default values.
    this->maximumIt = csParam.maxIt;
    this->currentIt = 0;
    this->currentThreshold = 0;
    this->nbThresholds = csParam.nbThresholds;
    this->decreasingFactor = csParam.expPercentageKept;
    this->runTime = csParam.maxRT;
    currentTemperature = (csParam.setupPercentage* initSol.getObjectiveValue()) / (-log(0.5));
}

ExponentialCoolingSchedule::~ExponentialCoolingSchedule() {
    // Nothing to be done
}

double ExponentialCoolingSchedule::getCurrentTemperature()
{
    currentIt++;
    clock_t currentTime = clock();
    double percentageTime = ((double)(currentTime - startingTime))/(endingTime- startingTime);
    double percentageIt = ((double)currentIt)/maximumIt;
    double percentageM = percentageTime;
    if(percentageIt > percentageM)
    {
        percentageM = percentageIt;
    }
    int aimedThreshold = (int)(percentageM*nbThresholds);
    while(currentThreshold < aimedThreshold)
    {
        currentThreshold++;
        currentTemperature *= decreasingFactor;
    }
    return currentTemperature;
}

void ExponentialCoolingSchedule::startSignal()
{
    startingTime = clock();
    endingTime = startingTime + CLOCKS_PER_SEC*runTime;
}