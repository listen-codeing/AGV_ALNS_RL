//
// Created by limin on 14/4/2025.
//

#ifndef AGV_CHARGING_TASK_H
#define AGV_CHARGING_TASK_H
#include <vector>

class Task {
public:
    enum TaskType {
        UNLOADING,  // ж������QC��YC (�Ӵ���ж�¼�װ��)
        LOADING     // װ������YC��QC (����װ��װ������)
    };
    void reset() {
        no = 0;
        startLoc = 0;
        endLoc = 0;
        distance = 0.0;
        isCharging = false;
        charingId = -1;
        tl = 0.0;
        te = 0.0;
        tr.clear();
        tf.clear();
        ce = 0.0;
        cl = 0.0;
        cr.clear();
        cf.clear();
        type = UNLOADING; // Ĭ������
    }

    int no;             // ������
    int startLoc;       // ��ʼλ��
    int endLoc;         // ����λ��
    double distance;    // ����
    bool isCharging;
    int charingId;  // ���վID������ǳ������
    double ready_time;

    // �����������ʱ��
    double tl;    // ����ʱ������
    double te;         // ����һ�������յ㵽��ǰ�������Ŀ�����ʻʱ��
    std::vector<double> tr;  // �ӵ�ǰ�����յ㵽�����վ�Ŀ�����ʻʱ��
    std::vector<double> tf;  // �Ӹ����վ����һ���������Ŀ�����ʻʱ��

    // ����ӵ��ܺ���ر���
    double cl;                    // ������ʻ�ܺ� c^l_{v,j}
    double ce;                    // ������ʻ�ܺģ�����ǰ������㣩c^u_{v,j}
    std::vector<double> cr;  // �ӵ�ǰ�����յ㵽�����վ�Ŀ�����ʻʱ��
    std::vector<double> cf;  // �Ӹ����վ����һ���������Ŀ�����ʻʱ��

    TaskType type;      // �������ͣ�װ����ж��

    Task() {};
    Task(int no, int startLoc, int endLoc, double tl)
            : no(no), startLoc(startLoc), endLoc(endLoc), tl(tl) {};
    Task(int no, int startLoc, int endLoc, double distance, double tl, TaskType type)
            : no(no), startLoc(startLoc), endLoc(endLoc), distance(distance), tl(tl), type(type) {};

    // ��ʼ�������վ��ʱ��������С
    void initialize_cs_times(int num_cs) {
        tr.resize(num_cs, 0.0);
        tf.resize(num_cs, 0.0);

        cr.resize(num_cs, 0.0);
        cf.resize(num_cs, 0.0);
    }

};


#endif //AGV_CHARGING_TASK_H
