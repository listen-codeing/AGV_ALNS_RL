//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_ISOLUTION_H
#define QC_AGV_ISOLUTION_H


#include <string>

class ISolution
{
public:
    virtual ~ISolution(){};

    virtual double getObjectiveValue()=0;

    virtual bool isFeasible()=0;

    //! �Ƚ���
    virtual bool operator<(ISolution& s) = 0;

    virtual ISolution* getCopy()=0;
    virtual ISolution* getDeepCopy()=0;
    virtual std::size_t getHash()=0;

};


#endif //QC_AGV_ISOLUTION_H
