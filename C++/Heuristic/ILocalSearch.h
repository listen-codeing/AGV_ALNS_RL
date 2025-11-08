//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_ILOCALSEARCH_H
#define QC_AGV_ILOCALSEARCH_H

#include <string>

class ISolution;

class ILocalSearch
{
public:
    //! Perform a local search on the solution.
    //! \return true if the solution is improved.
    virtual bool performLocalSearch(ISolution& sol)=0;

    //! \return the name of the local search operator.
    virtual std::string getName()=0;
};


#endif //QC_AGV_ILOCALSEARCH_H
