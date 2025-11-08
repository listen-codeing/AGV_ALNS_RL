//
// Created by limin on 13/4/2025.
//

#include "AGV.h"
#include "Parameter.h"
#include <cmath>
#include <random>

AGV::AGV(Parameter &pm, int id) {
    this->pm = &pm;
    this->truck_id = id;
    this->initial_soc = pm.varphies[id];
    this->task_sequence = pm.task_sequences[id];
}

void AGV::generate_task_sequence() {
    // 清空当前任务序列
    task_start_times.clear();
    task_end_times.clear();

    // 初始化其他任务相关向量
    task_start_times.resize(container_num, 0.0);
    task_end_times.resize(container_num, 0.0);

}
void AGV::reset() {
    pm = nullptr;
    truck_id = 0;
    initial_soc = 0.0;
    container_num = 0;

    task_sequence.clear();
    task_start_times.clear();
    task_end_times.clear();

    isCharge.clear();
    charging_sessions.clear();
    charging_start_times.clear();

    soc_after_task.clear();
    soc_before_task.clear();
    soc_at_cs_arrival.clear();
    soc_after_charging.clear();

    total_waiting_time = 0.0;
    total_charging_time = 0.0;
    total_travel_time = 0.0;

    original_Ts.clear();
    original_Te.clear();
    original_Se.clear();
    original_Sr.clear();
    original_Sf.clear();

    schedule_events.clear();
}

bool AGV::validate_schedule(const Parameter& params) {
    // 检查解决方案是否满足所有约束

    return true;
}

double AGV::get_travel_time() {
    this->total_travel_time = task_end_times[task_end_times.size()-1];
    return this->total_travel_time;
}

double AGV::get_charging_time() {
    this->total_charging_time = 0.0;
//    for (const auto& c_time : charging_durations) {
//        this->total_charging_time += c_time;
//    }
    return this->total_charging_time;
}

void AGV::update_schedule_from_event(int event_index, double new_start_time, double xi_uncertainty) {
    if (event_index >= schedule_events.size()) return;

    ScheduleEvent& event = schedule_events[event_index];

    // 计算时间偏移量
    double time_shift = new_start_time - event.start_time;

    // 如果没有时间变化，直接返回
    if (abs(time_shift) < 1e-6) return;

    // 更新当前事件
    event.start_time = new_start_time;

    if (event.type == ScheduleEventType::CHARGING) {
        event.end_time = new_start_time + pm->tu;
        // 重新计算等待时间（arrival_time不变，start_time变了）
        event.waiting_time = event.start_time - event.arrival_time;

        // 更新充电时间记录
        for (auto& charge_time : charging_start_times) {
            if (abs(charge_time - (new_start_time - time_shift)) < 1e-6) {
                charge_time = new_start_time;
                break;
            }
        }
    } else { // TASK
        event.end_time = new_start_time + task_sequence[event.task_id]->tl + xi_uncertainty;
        task_start_times[event.task_id] = event.start_time;
        task_end_times[event.task_id] = event.end_time;
    }

    // 优化：对后续所有事件进行简单时间平移
    for (int i = event_index + 1; i < schedule_events.size(); i++) {
        ScheduleEvent& next_event = schedule_events[i];

        // 简单平移所有时间
        next_event.start_time += time_shift;
        next_event.end_time += time_shift;

        if (next_event.type == ScheduleEventType::CHARGING) {
            next_event.arrival_time += time_shift;
            // 等待时间保持不变（start_time和arrival_time都平移了相同量）

            // 更新充电时间记录
            for (auto& charge_time : charging_start_times) {
                if (abs(charge_time - (next_event.start_time - time_shift)) < 1e-6) {
                    charge_time = next_event.start_time;
                    break;
                }
            }
        } else { // TASK
            // 更新任务时间数组
            task_start_times[next_event.task_id] = next_event.start_time;
            task_end_times[next_event.task_id] = next_event.end_time;
        }
    }
}

double AGV::calculate_remaining_time_from_event(int event_index) const {
    if (event_index >= schedule_events.size()) return 0.0;
    double start_time = schedule_events[event_index].start_time;
    double end_time = schedule_events.back().end_time;
    return end_time - start_time;
}

double AGV::get_makespan() const {
    if (schedule_events.empty()) return 0.0;
    return schedule_events.back().end_time;
}