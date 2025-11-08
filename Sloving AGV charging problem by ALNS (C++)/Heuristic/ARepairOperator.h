//
// Created by limin on 2023/10/13.
//

#ifndef QC_AGV_AREPAIROPERATOR_H
#define QC_AGV_AREPAIROPERATOR_H

#include "AOperator.h"

class ISolution;

class ARepairOperator : public AOperator {

public:
    ARepairOperator(std::string s) : AOperator(s)
    {
    }

    virtual ~ARepairOperator(){};

    virtual void repairSolution(ISolution& sol)=0;
};


#endif //QC_AGV_AREPAIROPERATOR_H
