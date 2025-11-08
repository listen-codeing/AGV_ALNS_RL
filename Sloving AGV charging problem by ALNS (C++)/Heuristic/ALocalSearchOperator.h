//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_ALOCALSEARCHOPERATOR_H
#define QC_AGV_ALOCALSEARCHOPERATOR_H


#include "AOperator.h"
#include "../AGV.h"

class ISolution;
class Parameter;

class ALocalSearchOperator : public AOperator {

public:
    ALocalSearchOperator(std::string s) : AOperator(s)
    {
    }

    virtual ~ALocalSearchOperator(){};

    virtual void localSearchSolution(Parameter *pm, AGV *truck, ISolution &sol) =0;

};

#endif //QC_AGV_ALOCALSEARCHOPERATOR_H
