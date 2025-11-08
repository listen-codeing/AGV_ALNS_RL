//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_IACCEPTANCEMODULE_H
#define QC_AGV_IACCEPTANCEMODULE_H


#include <iostream>
#include "ISolution.h"
#include "ALNS_Iteration_Status.h"
#include "IBestSolutionManager.h"

/*!
 * \class IAcceptanceModule.
 * \brief This is an interface to define acceptance modules within the ALNS.
 */

class IAcceptanceModule
{
public:
    // 接受新解的标准
    virtual bool transitionAccepted(IBestSolutionManager& bestSolutionManager, ISolution& currentSolution, ISolution& newSolution, ALNS_Iteration_Status& status) = 0;

    //! Some Acceptance modules needs to initialize some variable
    //! only when the solver actualy starts working. In this case
    //! you should override this method.
    virtual void startSignal(){};
};


#endif //QC_AGV_IACCEPTANCEMODULE_H
