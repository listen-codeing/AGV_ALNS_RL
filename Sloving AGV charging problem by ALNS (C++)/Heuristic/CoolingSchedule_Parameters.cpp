//
// Created by limin on 2023/9/7.
//

#include "CoolingSchedule_Parameters.h"
#include "ALNS_Parameters.h"
#include <assert.h>


using namespace std;

CoolingSchedule_Parameters::CoolingSchedule_Parameters(ALNS_Parameters& alnsPara)
{
    kind = Exponential_mix;

    expPercentageKept = 0.999;

    setupPercentage = 0.1;

    nbThresholds = 1000;

    maxIt = alnsPara.getMaxNbIterations();

    maxRT = alnsPara.getMaxRunningTime();
}

CoolingSchedule_Parameters::~CoolingSchedule_Parameters()
{
    // Nothing to be done.
}

void CoolingSchedule_Parameters::sanityChecks()
{
    //TODO develop sanity checks.
}

CoolingSchedule_Parameters::CoolingSchedule_Parameters(
        CoolingSchedule_Parameters& p)
{
    kind = p.kind;

    expPercentageKept = p.expPercentageKept;

    setupPercentage = p.setupPercentage;

    nbThresholds = p.nbThresholds;

    maxIt = p.maxIt;

    maxRT = p.maxRT;
}