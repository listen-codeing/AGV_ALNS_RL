//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_COOLINGSCHEDULE_PARAMETERS_H
#define QC_AGV_COOLINGSCHEDULE_PARAMETERS_H


#include <cstring>
#include <iostream>

class ALNS_Parameters;

/*!
 * \class CoolingSchedule_Parameters.
 * \brief This class represent parameters to be used to instantiate cooling schedules.
 */

class CoolingSchedule_Parameters
{
public:

    //! Enumeration representing the various kind of cooling schedules that can be handled.
    enum CSKind{
        Linear_it,
        Linear_time,
        Linear_mix,
        Exponential_it,
        Exponential_time,
        Exponential_mix
    };

    //! Constructor.
    CoolingSchedule_Parameters(ALNS_Parameters& alnsParam);

    //! Default Constructor.
    CoolingSchedule_Parameters();

    //! Copy constructor.
    CoolingSchedule_Parameters(CoolingSchedule_Parameters& p);

    //! Destructor.
    ~CoolingSchedule_Parameters();

    //! Perform some sanity checks on the values of the parameters.
    void sanityChecks();

    //! Load the parameters from a text file.
    //! \param path the route to the text file containing the parameters.
    void loadParameters(std::string path);

    //! Load the parameters from a text file.
    //! \param path the route to the text file containing the parameters.
    void loadXMLParameters(std::string path);

    //! The kind of cooling schedule to be used.
    CSKind kind;

    //! In case of the use of an exponential cooling schedule, the percentage of
    //! the current temperature kept at each temperature recomputation.
    double expPercentageKept;

    //! To determine the starting temperature, how worse than the initial solution can,设定最开始能接受解超过初始解的百分比,比如0.05
    //! a solution be to have a 50% chance of being accepted at the starting temperature.
    double setupPercentage;

    //! The number of temperature recomputations during the optimization process.
    size_t nbThresholds;

    //! The maximum number of iterations.
    size_t maxIt;

    //! The maximum running time.
    size_t maxRT;


};

#endif //QC_AGV_COOLINGSCHEDULE_PARAMETERS_H
