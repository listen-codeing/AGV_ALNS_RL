//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_LINEARCOOLINGSCHEDULE_H
#define QC_AGV_LINEARCOOLINGSCHEDULE_H


#include <cstring>
#include "ICoolingSchedule.h"


class ISolution;
class CoolingSchedule_Parameters;

class LinearCoolingSchedule: public ICoolingSchedule {
private:
    //! The current temperature.
    double currentTemperature;

    //! The amount to removeTask_basedTaskIndex at each temperature recomputation.
    double amountRemove;
public:
    //! Constructor.
    //! \param initSol the initial solution.
    //! \param csParam the cooling schedule parameters.
    //! \param nbIterations the number of iterations to be performed.
    LinearCoolingSchedule(ISolution& initSol, CoolingSchedule_Parameters& csParam, size_t nbIterations);

    //! Constructor.
    //! \param startingTemperature the initial temperature.
    //! \param nbIterations the number of iterations to be performed.
    LinearCoolingSchedule(double startingTemperature, size_t nbIterations);

    //! Destructor.
    virtual ~LinearCoolingSchedule();

    //! Compute and return the current temperature.
    //! \return the current temperature.
    double getCurrentTemperature();

    void startSignal(){};
};
#endif //QC_AGV_LINEARCOOLINGSCHEDULE_H
