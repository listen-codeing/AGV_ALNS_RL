//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_STATISTICS_H
#define QC_AGV_STATISTICS_H


#include <vector>
#include <time.h>
#include <iostream>

class Statistics {
public:
    //! Constructor.
    Statistics(){};

    //! Destructor.
    virtual ~Statistics();

    //! This method adds an entry to the data
    void addEntry(double timeStamp,
                  size_t iteration,
                  std::string destroyName,
                  std::string recreateName,
                  double newCost,
                  double currentCost,
                  double bestCost,
                  int cumKS);

    void addOperatorEntry(std::vector<double>* weight,
                          std::vector<size_t>* calls);

    void addOperatorsNames(std::vector<std::string>* names){operatorNames = names;};

    //! This method generate the file containing the datas.
    void generateStatsFile(std::string path, std::string pathOp);

    void setStart(){start = clock();};

private:
    std::vector<double> timeStamps;
    std::vector<size_t> iterations;
    std::vector<std::string> destroyNames;
    std::vector<std::string> recreateNames;
    std::vector<double> newCosts;
    std::vector<double> currentCosts;
    std::vector<double> bestCosts;
    std::vector<int> cumulativeKnownSolutions;
    std::vector<std::vector<double>* > weights;
    std::vector<std::vector<size_t>* > nbCalls;
    std::vector<double> timeStampsOperators;
    std::vector<std::string>* operatorNames;
    clock_t start;

};


#endif //QC_AGV_STATISTICS_H
