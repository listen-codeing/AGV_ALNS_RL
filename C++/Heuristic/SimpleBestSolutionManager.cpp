//
// Created by limin on 2023/9/7.
//

#include "SimpleBestSolutionManager.h"
#include <list>
#include "ISolution.h"
#include "ALNS_Iteration_Status.h"
#include "ALNS_Parameters.h"
#include "AGV_solution.h"

using namespace std;

SimpleBestSolutionManager::SimpleBestSolutionManager(ALNS_Parameters& param) {
    parameters = &param;
}

SimpleBestSolutionManager::~SimpleBestSolutionManager() {
    for(list<ISolution*>::iterator it = bestSols.begin(); it != bestSols.end(); it++)
    {
        delete (*it);
    }
}

bool SimpleBestSolutionManager::isNewBestSolution(ISolution& sol)
{
    // ????????????н?
    if (!sol.isFeasible()){
        return false;
    }else{
        //???sol??????????е????????????????false??
        //???sol????????????????е?????????????false??
        //???sol??????<????????е????????????????????????????????????
        //????????з???false????sol????????????
        for(list<ISolution*>::iterator it = bestSols.begin(); it != bestSols.end(); it++)
        {
            ISolution& currentSol = *(*it);
            if(currentSol<sol)
            {
                return false;
            }
            else if(sol<currentSol)
            {
                delete *it;
                it = bestSols.erase(it);
                if(it == bestSols.end())
                {
                    break;
                }
            }
            else if(currentSol.getHash() == sol.getHash())
            {
                return false;
            }
        }
        ISolution* copy = sol.getDeepCopy();
        bestSols.push_back(copy);
        return true;
    }
}



ISolution* SimpleBestSolutionManager::reloadBestSolution(ISolution* currSol, ALNS_Iteration_Status& status)
{
    if(status.getNbIterationWithoutImprovementSinceLastReload() > 0 &&
       ((status.getNbIterationWithoutImprovementSinceLastReload() % parameters->getReloadFrequency()) == 0))
    {
        status.setNbIterationWithoutImprovementSinceLastReload(0);
        delete currSol;
        return getBestSol();
    }
    else
    {
        return currSol;
    }
}

ISolution* SimpleBestSolutionManager::getBestSol() {
    return bestSols.back()->getCopy();
}