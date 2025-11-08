//
// Created by limin on 13/4/2025.
//

#ifndef AGV_CHARGING_AGV_NEW_H
#define AGV_CHARGING_AGV_NEW_H

#include <vector>
#include <utility> // for std::pair

class Parameter;
class Task;

enum class ScheduleEventType {
    TASK,           // ��ҵ����
    CHARGING        // �������
};

// ����ScheduleEvent���캯��
struct ScheduleEvent {
    ScheduleEventType type;
    int task_id;
    double start_time;
    double end_time;
    double arrival_time;
    double waiting_time;

    // �����¼����캯��
    ScheduleEvent(ScheduleEventType t, int id, double start, double end)
            : type(t), task_id(id), start_time(start), end_time(end),
              arrival_time(start), waiting_time(0.0) {}

    // ����¼����캯��
    ScheduleEvent(ScheduleEventType t, int id, double arrival, double start, double end)
            : type(t), task_id(id), start_time(start), end_time(end),
              arrival_time(arrival), waiting_time(start - arrival) {}

    // ���µȴ�ʱ��ķ���
    void update_waiting_time() {
        if (type == ScheduleEventType::CHARGING) {
            waiting_time = start_time - arrival_time;
        }
    }
};

class AGV {

public:
    Parameter* pm;

    int truck_id;                  // AGV���
    double initial_soc;          // ��ʼ��ص���
    int container_num;           // ��װ������

    // �������
    std::vector<Task*> task_sequence;      // ��������
    // ÿ������Ľ���ʱ��
    std::vector<double> task_arrival_times;    // ÿ������Ŀ�ʼʱ��
    std::vector<double> task_start_times;    // ÿ������Ŀ�ʼʱ��
    std::vector<double> task_end_times;      // ÿ������Ľ���ʱ��
    std::vector<double> task_completion_times;      // �������ʱ�䣨��������ȷ���ԣ�



    // ������
    std::vector<bool> isCharge;     // ÿ��������Ƿ���
    std::vector<std::pair<int, int>> charging_sessions;  // ���Ự(���վ���,�Ự���)
    std::vector<double> charging_start_times; // ��翪ʼʱ��
    std::vector<double> arrival_at_station; // ������վ��ʱ��
    std::vector<double> charging_end_times; // ��翪ʼʱ��
    std::vector<double> charging_durations;   // ������ʱ��

    // ���״̬
    std::vector<double> soc_after_task;       // ÿ�������ĵ�ص���
    std::vector<double> soc_before_task;       // ÿ������ʼǰ�ĵ�ص���
    std::vector<double> soc_at_cs_arrival;    // ������վʱ�ĵ�ص���
    std::vector<double> soc_after_charging;   // ����ĵ�ص���

    std::vector<double> soc_charging_durations;   // ���˶��ٵ�

    // ͳ����Ϣ
    double max_completion_time;
    double total_waiting_time;    //
    double total_charging_time;    //
    double total_travel_time;

    // ԭʼ Gurobi ����ֵ�����ã�
    std::vector<double> original_Ts;    // ԭʼ����ʼʱ�����ֵ
    std::vector<double> original_Te;    // ԭʼ�������ʱ�����ֵ
    std::vector<double> original_Se;    // ԭʼ����� SOC ����ֵ
    std::vector<double> original_Sr;    // ԭʼ������վʱ SOC ����ֵ
    std::vector<double> original_Sf;    // ԭʼ���� SOC ����ֵ

    // ���캯��
    AGV() = default;
    AGV(Parameter &pm, int id);


    bool validate_schedule(const Parameter& params); // ��֤��������Ƿ���������Լ��

    void generate_task_sequence();

    double get_travel_time();

    double get_charging_time();

    // ͳһ�ĵ����¼�����
    std::vector<ScheduleEvent> schedule_events;

    void update_schedule_from_event(int event_index, double new_start_time, double xi_uncertainty);

    double calculate_remaining_time_from_event(int event_index) const;
    double get_makespan() const;
    void reset();

};


#endif //AGV_CHARGING_AGV_NEW_H
