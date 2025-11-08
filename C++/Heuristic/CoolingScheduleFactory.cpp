//
// Created by limin on 2023/9/7.
//

#include "CoolingScheduleFactory.h"
#include "CoolingSchedule_Parameters.h"
#include "ICoolingSchedule.h"
#include "LinearCoolingSchedule.h"
#include "ExponentialCoolingSchedule.h"

ICoolingSchedule* CoolingScheduleFactory::makeCoolingSchedule(ISolution& sol,
                                                              CoolingSchedule_Parameters& param)
{
    // Depending of the value of the field kind
    // of the cooling schedule parameters
    // a corresponding cooling schedule is build.
    if(param.kind == CoolingSchedule_Parameters::Linear_it)
    {
        std::cout << "LI" << std::endl;
        return dynamic_cast<ICoolingSchedule*>(new LinearCoolingSchedule(sol,param,param.maxIt));
    }
//    else if(param.kind == CoolingSchedule_Parameters::Linear_time)
//    {
//        std::cout << "LT" << std::endl;
//        return dynamic_cast<ICoolingSchedule*>(new TimeLinearCoolingSchedule(sol,param));
//    }
//    else if(param.kind == CoolingSchedule_Parameters::Linear_mix)
//    {
//        std::cout << "LM" << std::endl;
//        return dynamic_cast<ICoolingSchedule*>(new MixLinearCoolingSchedule(sol,param));
//    }
    else if(param.kind == CoolingSchedule_Parameters::Exponential_mix)
    {
//        std::cout << "EXP" << std::endl;
        return dynamic_cast<ICoolingSchedule*>(new ExponentialCoolingSchedule(sol,param));
    }
    else
    {
        std::cerr << "This cooling schedule is not available yet" << std::endl;
        exit(EXIT_FAILURE);
    }
}