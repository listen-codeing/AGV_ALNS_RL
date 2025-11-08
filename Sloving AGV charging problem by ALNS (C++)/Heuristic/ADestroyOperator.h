//
// Created by limin on 2023/10/13.
//

#ifndef QC_AGV_ADESTROYOPERATOR_H
#define QC_AGV_ADESTROYOPERATOR_H

#include "AOperator.h"

class ISolution;

class ADestroyOperator : public AOperator{

protected:
    // //! The minimum destroy size used.
    // size_t minimunDestroy;
    // //! The maximum destroy size used.
    // size_t maximumDestroy;
    double minDestroyPerc;
    double maxDestroyPerc;

public:
    //! Constructor.
    //! \param mini the minimum destroy size.
    //! \param maxi the maximum destroy size.
    //! \param s the name of the destroy operator.
    // ADestroyOperator(size_t mini, size_t maxi, std::string s) : AOperator(s)
    // {
    //     minimunDestroy = mini;
    //     maximumDestroy = maxi;
    // }
    ADestroyOperator(double mini, double maxi, std::string s) : AOperator(s)
    {
        minDestroyPerc = mini;
        maxDestroyPerc = maxi;
    }

    //! Destructor.
    virtual ~ADestroyOperator(){};

    //! This function is the one called to destroy a solution.
    //! \param sol the solution to be destroyed.
    // 虚函数直接=0，表示子类必须重写
    virtual void destroySolution(ISolution& sol)=0;
};


#endif //QC_AGV_ADESTROYOPERATOR_H
