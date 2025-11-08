//
// Created by limin on 2023/9/7.
//

#ifndef QC_AGV_ALNS_PARAMETERS_H
#define QC_AGV_ALNS_PARAMETERS_H


#include <assert.h>
#include <cstring>
#include <iostream>
#include <vector>

class Parameter;
class AGV;

class ALNS_Parameters
{

public:
    Parameter* pm;
    //AGV* truck;
    std::vector<std::shared_ptr<AGV>> trucks;  // 替换原来的 AGV* truck;

    //! ��ͬ����ֹ����,�������������������ʱ�䣬����������û��������һ�ֻ�ϲ���
    enum StoppingCriteria {
        MAX_IT,
        MAX_RT,
        MAX_IT_NO_IMP,
        ALL
    };

    enum AcceptanceCriterioKind {
        SA
    };

    //! Constructor.
    //ALNS_Parameters(Parameter* pm, AGV* truck);
    ALNS_Parameters(Parameter* pm, const std::vector<std::shared_ptr<AGV>>& trucks);


    //! Destructor.
    ~ALNS_Parameters();

    size_t getReloadFrequency() const
    {
        return reloadFrequency;
    }

    void setReloadFrequency(size_t reloadFrequency)
    {
        assert(!lock);
        this->reloadFrequency = reloadFrequency;
    }

    //! Simple getter.
    std::vector<std::string> getForbidenLsOperators() const
    {
        return forbidenLsOperators;
    }

    void addForbiddenLsOperator(std::string lsOp)
    {
        forbidenLsOperators.push_back(lsOp);
    }

    //! Simple getter.
    std::vector<std::string> getForbidenOperators() const
    {
        return forbidenOperators;
    }

    void addForbiddenOperator(std::string op)
    {
        forbidenOperators.push_back(op);
    }

    bool getPerformLocalSearch() const
    {
        return performLocalSearch;
    }

    void setPerformLocalSearch(bool performLocalSearch)
    {
        assert(!lock);
        this->performLocalSearch = performLocalSearch;
    }

    int getLogFrequency() const
    {
        return logFrequency;
    }

    void setLogFrequency(int logFrequency)
    {
        assert(!lock);
        this->logFrequency = logFrequency;
    }

    std::string getStatsGlobPath() const
    {
        return statsGlobPath;
    }

    std::string getStatsOpPath() const
    {
        return statsOpPath;
    }

    void setStatsGlobPath(std::string statsGlobPath)
    {
        this->statsGlobPath = statsGlobPath;
    }

    void setStatsOpPath(std::string statsOpPath)
    {
        this->statsOpPath = statsOpPath;
    }

    //! Simple getter.
    AcceptanceCriterioKind getAcKind() const
    {
        return acKind;
    }

    //! Simple getter.
    std::string getAcPath() const
    {
        return acPath;
    }

    //! Simple getter.
    void setAcKind(AcceptanceCriterioKind acKind)
    {
        assert(!lock);
        this->acKind = acKind;
    }

    //! Simple getter.
    void setAcPath(std::string acPath)
    {
        assert(!lock);
        this->acPath = acPath;
    }

    //! Simple getter.
    double getProbabilityOfNoise() const
    {
        return probabilityOfNoise;
    }

    //! Simple getter.
    void setProbabilityOfNoise(double probabilityOfNoise)
    {
        assert(!lock);
        this->probabilityOfNoise = probabilityOfNoise;
    }

    //! Simple getter.
    size_t getMaxNbIterations() const
    {
        return maxNbIterations;
    }

    //! Simple getter.
    size_t getMaxNbIterationsNoImp() const
    {
        return maxNbIterationsNoImp;
    }

    //! Simple getter.
    double getMaxRunningTime() const
    {
        return maxRunningTime;
    }

    //! Simple getter.
    double getMaximumWeight() const
    {
        return maximumWeight;
    }

    //! Simple getter.
    double getMinimumWeight() const
    {
        return minimumWeight;
    }

    //! Simple getter.
    size_t getNbItBeforeReinit() const
    {
        return nbItBeforeReinit;
    }

    //! Simple getter.
    bool getNoise() const
    {
        return noise;
    }

    //! Simple getter.
    double getRho() const
    {
        return rho;
    }

    //! Simple getter.
    int getSigma1() const
    {
        return sigma1;
    }

    //! Simple getter.
    int getSigma2() const
    {
        return sigma2;
    }

    //! Simple getter.
    int getSigma3() const
    {
        return sigma3;
    }

    //! Simple getter.
    int getsimilarity1() const
    {
        return similarity1;
    }
    int getsimilarity2() const
    {
        return similarity2;
    }

    //! Simple getter.
    StoppingCriteria getStopCrit() const
    {
        return stopCrit;
    }

    //! Simple getter.
    size_t getTimeSegmentsIt() const
    {
        return timeSegmentsIt;
    }

    //! Simple setter.
    void setMaxNbIterations(size_t maxNbIterations)
    {
        assert(!lock);
        this->maxNbIterations = maxNbIterations;
    }

    //! Simple setter.
    void setMaxNbIterationsNoImp(size_t maxNbIterationsNoImp)
    {
        assert(!lock);
        this->maxNbIterationsNoImp = maxNbIterationsNoImp;
    }

    //! Simple setter.
    void setMaxRunningTime(double maxRunningTime)
    {
        assert(!lock);
        this->maxRunningTime = maxRunningTime;
    }

    //! Simple setter.
    void setMaximumWeight(double maximumWeight)
    {
        assert(!lock);
        this->maximumWeight = maximumWeight;
    }

    //! Simple setter.
    void setMinimumWeight(double minimumWeight)
    {
        assert(!lock);
        this->minimumWeight = minimumWeight;
    }

    //! Simple setter.
    void setNbItBeforeReinit(size_t nbItBeforeReinit)
    {
        assert(!lock);
        this->nbItBeforeReinit = nbItBeforeReinit;
    }

    //! Simple setter.
    void setNoise(bool noise)
    {
        assert(!lock);
        this->noise = noise;
    }

    //! Simple setter.
    void setRho(double rho)
    {
        assert(!lock);
        this->rho = rho;
    }

    //! Simple setter.
    void setSigma1(int sigma1)
    {
        assert(!lock);
        this->sigma1 = sigma1;
    }

    //! Simple setter.
    void setSigma2(int sigma2)
    {
        assert(!lock);
        this->sigma2 = sigma2;
    }

    //! Simple setter.
    void setSigma3(int sigma3)
    {
        assert(!lock);
        this->sigma3 = sigma3;
    }

    //! Simple setter.
    void setStopCrit(StoppingCriteria stopCrit)
    {
        assert(!lock);
        this->stopCrit = stopCrit;
    }

    //! Simple setter.
    void setTimeSegmentsIt(size_t timeSegmentsIt)
    {
        assert(!lock);
        this->timeSegmentsIt = timeSegmentsIt;
    }

    //! Simple setter.
    void setLock()
    {
        lock = true;
    }

    int getMaxDestroyNum() const {
        return maximumDestroy;
    }

    void setMaxDestroyPerc(int maxDestroyPerc) {
        this->maxDestroyPerc = maxDestroyPerc;
    }

    int getMinDestroyNum() const {
        return minimunDestroy;
    }

    void setMinDestroyPerc(int minDestroyPerc) {
        this->minDestroyPerc = minDestroyPerc;
    }

    double getMinDestroyPerc() const {
        return minDestroyPerc;
    }
    double getMaxDestroyPerc() const {
        return maxDestroyPerc;
    }

protected:
    size_t maximumDestroy;
    size_t minimunDestroy;

    // ����������
    size_t maxNbIterations;

    // �������ʱ��
    double maxRunningTime;


    // û������������������
    size_t maxNbIterationsNoImp;

    // ֹͣ����
    StoppingCriteria stopCrit;

    // �Ƿ����noise��׼
    bool noise;

    // ���¼������ӷ����Ĵ���
    size_t timeSegmentsIt;

    // ���³�ʼ�����ӷ����Ĵ���
    size_t nbItBeforeReinit;

    // ���ȫ�����Ž⣬�ӷ�
    int sigma1;

    // �������·������·�����Ŀ��ֵ�ȵ�ǰ������Ŀ��ֵ���ã��ӷ�
    int sigma2;

    // �������·������·�����Ŀ��ֵ�ȵ�ǰ������Ŀ��ֵ����ӷ�
    int sigma3;

    // ���ȫ�����Ž⣬�ӷ�
    int similarity1;

    // �������·������·�����Ŀ��ֵ�ȵ�ǰ������Ŀ��ֵ���ã��ӷ�
    int similarity2;

    //! reaction factor 0 <= rho <= 1 for the update of the weights of the
    //! operators. ���²���
    double rho;

    // ����Ȩ���½�
    double minimumWeight;

    // ����Ȩ���Ͻ�
    double maximumWeight;

    // ���������ӵĸ���
    double probabilityOfNoise;

    // ����׼��
    AcceptanceCriterioKind acKind;

    // ����·��
    std::string acPath;

    // ���Ž��ļ����·��
    std::string statsGlobPath;

    // ͳ���ļ����·��
    std::string statsOpPath;

    // ÿ�ε�����־���Ƶ��
    int logFrequency;

    // ���ɱ�������
    std::vector<std::string> forbidenOperators;
    // ���ɱ����ֲ���������
    std::vector<std::string> forbidenLsOperators;

    // ʹ���ƻ����ӵ���С����
    double minDestroyPerc;

    // ʹ���ƻ����ӵ�������
    double maxDestroyPerc;

    // ���¼������Ž��Ƶ��
    size_t reloadFrequency;

    // �Ƿ��þֲ�����
    bool performLocalSearch;

    // �������ܣ��Ż���ʼ֮�󣬲����������ò���
    bool lock;

};

#endif //QC_AGV_ALNS_PARAMETERS_H
