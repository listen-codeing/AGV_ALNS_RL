//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_AOPERATOR_H
#define QC_AGV_AOPERATOR_H

#include <iostream>

class AOperator
{
private:
    // 使用总数
    size_t totalNumberOfCalls;

    //上次使用后的使用数
    size_t nbCallsSinceLastEval;

    // 得分
    double score;

    //权重
    double weight;

    // 名称
    std::string operatorName;

protected:
    // 是否使用noise模式.
    bool noise;
public:

    //! Constructor.
    AOperator(std::string name){
        operatorName = name;
        init();
    }

    //! Destructor.
    virtual ~AOperator(){};

    // 初始化过程
    void init()
    {
        totalNumberOfCalls = 0;
        nbCallsSinceLastEval = 0;
        score = 0;
        weight = 1;
    }

    //重置使用次数.
    void resetNumberOfCalls()
    {
        nbCallsSinceLastEval = 0;
    }

    size_t getTotalNumberOfCalls(){return totalNumberOfCalls;};

    size_t getNumberOfCallsSinceLastEvaluation(){return nbCallsSinceLastEval;};

    void increaseNumberOfCalls()
    {
        totalNumberOfCalls++;
        nbCallsSinceLastEval++;
    }

    double getScore() const
    {
        return score;
    }

    double getWeight() const
    {
        return weight;
    }

    void resetScore()
    {
        this->score = 0;
    }

    void setScore(double s)
    {
        this->score = s;
    }

    void setWeight(double weight)
    {
        this->weight = weight;
    }

    std::string getName(){return operatorName;};

    void setNoise(){noise=true;};

    void unsetNoise(){noise=false;};

};


#endif //QC_AGV_AOPERATOR_H
