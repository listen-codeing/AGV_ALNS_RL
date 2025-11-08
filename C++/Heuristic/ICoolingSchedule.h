//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_ICOOLINGSCHEDULE_H
#define QC_AGV_ICOOLINGSCHEDULE_H


class ICoolingSchedule
{
public:
    //! \return the current temperature.
    virtual double getCurrentTemperature()=0;

    //! This method should be called when the optimization
    //! process start. The cooling schedules that actually need
    //! this should override this method.
    virtual void startSignal(){};
};

#endif //QC_AGV_ICOOLINGSCHEDULE_H
