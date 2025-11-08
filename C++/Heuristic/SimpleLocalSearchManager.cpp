//
// Created by limin on 2023/9/7.
//

#include "SimpleLocalSearchManager.h"
#include "ALNS_Parameters.h"
#include "ALNS_Iteration_Status.h"
#include "ALocalSearchOperator.h"
#include "ISolution.h"
#include "ILocalSearch.h"


bool SimpleLocalSearchManager::useLocalSearch(ISolution& sol, ALNS_Iteration_Status& status)
{
    if(status.getNewBestSolution()!=ALNS_Iteration_Status::TRUE
       || status.getAcceptedAsCurrentSolution()!=ALNS_Iteration_Status::UNKNOWN)
    {
        return false;
    }
    else
    {
        status.setLocalSearchUsed(ALNS_Iteration_Status::TRUE);
        bool improvement;
        for(size_t i = 0; i < localSearchOperators.size(); i++)
        {
            improvement = localSearchOperators[i]->performLocalSearch(sol) ;
            if (improvement){
                break;
            }
        }
        if(improvement)
        {
            status.setImproveByLocalSearch(ALNS_Iteration_Status::TRUE);
            return true;
        }
        else
        {
            status.setImproveByLocalSearch(ALNS_Iteration_Status::FALSE);
            return false;
        }
    }
}

void SimpleLocalSearchManager::addLocalSearchOperator(ILocalSearch& ls)
{
    //TODO find out why the set.find() == set.end() does not work.
    bool ok = true;
    for(size_t i=0; i< param->getForbidenLsOperators().size() && ok; i++)
    {
        if(param->getForbidenLsOperators()[i] == ls.getName())
        {
            std::cout << "NO " << ls.getName() << std::endl;
            ok = false;
        }
    }
    if(ok)
    {
        localSearchOperators.push_back(&ls);
    }

}