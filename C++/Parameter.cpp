//
// Created by limin on 13/4/2025.
//

#include <iomanip>
#include "Parameter.h"
#include "TestUnit.h"
#include "random_seed.h"

Parameter::Parameter(TestUnit *tm, TestCase testCase)
{
    this->tm = tm;
    this->isPrint = !tm->Totxt;
    this->wassterTemp = this->Theta / this->epsilon;
    J = testCase.container_num;
    V = testCase.truck_num;
    C = testCase.charge_station_num;
    generate_distance_matrix();
    this->omega_num = tm->omega_num;
    process_precision();
    //?????????? (J) ?????????????????? (V) ??AGV???????????????
    //???????AGV?????????????????? J_v??
    generate_agv_container_num();
    generate_tasks();           // ????????
    generate_travel_time_scenarios(); // ??????
    generate_energy_uncertainty_scenarios();

    assign_tasks_to_trucks();     // ?????????AGVs
    generate_all_task();     // ?????????????????
    if (isPrint) print_tasks();              // ??????????
//    pm.save_tasks_to_file("tasks.csv"); // ???浽???

    calculate_time_dro();
    calculate_zeta_dro();
    generate_agv_init_SOC();
    read_charge_table_from_csv();

}

void Parameter::generate_agv_container_num() {
    // ?????J_v????
    J_v.resize(V, 0);

    // ????????????????
    int base_tasks_per_agv = J / V;
    int remaining_tasks = J % V;

    // ??????????????????
    for (int i = 0; i < V; i++) {
        J_v[i] = base_tasks_per_agv;
    }

    // ????????????
    // ??ù???????????????????????
    std::vector<int> indices(V);
    std::iota(indices.begin(), indices.end(), 0); // ???0,1,...,V-1
    //sharedGenerator.seed(1);
    // ???sharedGenerator??????std::random_device
    std::shuffle(indices.begin(), indices.end(), sharedGenerator);

    // ??????????????????ЩAGV
    for (int i = 0; i < remaining_tasks; i++) {
        J_v[indices[i]]++;
    }

    // ????????????? (???)
    // ??????????????????????????????????AGV????????
    int num_adjustments = V / 2; // ????????????????????????

    for (int i = 0; i < num_adjustments; i++) {
        // ???sharedGenerator???????????????????rand()
        std::uniform_int_distribution<int> dist(0, V-1);
        int idx1 = dist(sharedGenerator);
        int idx2 = dist(sharedGenerator);

        // ???idx1??idx2???
        while (idx1 == idx2) {
            idx2 = dist(sharedGenerator);
        }

        // ???AGV?????????????е???
        if (J_v[idx1] > 1) {
            J_v[idx1]--;
            J_v[idx2]++;
        }
    }

    // ?????????????????
    int sum = 0;
    for (int i = 0; i < V; i++) {
        sum += J_v[i];
    }

    // ??????????????????
    if (sum != J) {
        int diff = J - sum;

        std::uniform_int_distribution<int> dist(0, V-1);
        int idx = dist(sharedGenerator);

        J_v[idx] += diff;
    }
}


void Parameter::read_distance_from_csv() {
    try {
        std::ifstream file(this->tm->fileName_distance);
        if (!file.is_open()) {
            std::cerr << "??????????????" << std::endl;
            return;
        }

        std::string line;
        int row = 0;

        // ??????????
        std::getline(file, line);
        std::getline(file, line);

        // ?????????
        int total_nodes = 1 + 5 + 10 + 8; // O + QCs + YCs + CSs

        // ????????????
        distance.resize(total_nodes, std::vector<double>(total_nodes, -1.0));

        while (std::getline(file, line) && row < total_nodes) {
            std::stringstream ss(line);
            std::string cell;
            int col = 0;

            // ????????У?????????
            std::getline(ss, cell, ',');

            while (std::getline(ss, cell, ',') && col < total_nodes) {
                // ???????"-"?-1
                if (cell.empty() || cell == "-") {
                    distance[row][col] = -1.0;
                } else {
                    try {
                        distance[row][col] = std::stod(cell)/2;
                    } catch (const std::invalid_argument&) {
                        distance[row][col] = -1.0;
                    }
                }
                col++;
            }
            row++;
        }

        file.close();
        //        std::cout << "??????????CSV??????????" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "???CSV?????????: " << e.what() << std::endl;
    }
}

void Parameter::read_charge_table_from_csv() {
    charge_table.clear();
    std::ifstream file(this->tm->fileName_SOC);
    if (!file.is_open()) {
        std::cout << "???δ??" << std::endl;
        return;
    }
    std::string line;
    // ???????
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        if (row.size() < 6) continue; // ????????????

        ChargeData data;
        data.no        = std::stod(row[0]);
        data.soc_start = std::stod(row[1]);
        data.soc_end   = std::stod(row[2]);
        data.soc_delta = std::stod(row[3]);
        data.time_min  = std::stod(row[4]);
        data.time_hour = std::stod(row[5]);
        charge_table.push_back(data);
    }
}

//double Parameter::get_charge_time_min(double true_soc_start, double true_soc_end) {
//    if (true_soc_start>=true_soc_end) {
//        return 0;
//    }
//    if (true_soc_start<0||true_soc_end>Pi+1e-3) {
//        return -1;
//    }
//
//    for (const auto& data : charge_table) {
//
//        if (data.soc_start == static_cast<int>(std::round(true_soc_start)) &&
//            data.soc_end == static_cast<int>(std::round(true_soc_end))) {
//            return data.time_min;
//            }
//    }
//    // cout<<"static_cast<int>(std::round(100*true_soc_start/Pi)):"<<static_cast<int>(std::round(100*true_soc_start/Pi))<<endl;
//    // cout<<"static_cast<int>(std::round(100*true_soc_end/Pi)):"<<static_cast<int>(std::round(100*true_soc_end/Pi))<<endl;
//    // cout<<"true_soc_start:"<<true_soc_start<<endl;
//    // cout<<"true_soc_end:"<<true_soc_end<<endl;
//    // cout<<"??????"<<endl;
//    // system("pause");
//    // ??????????-1???????
//    return -1.0;
//}
double Parameter::get_charge_time_min(double true_soc_start, double true_soc_end) {
    if(tm->warm_start)
    {
        if (true_soc_start>=true_soc_end) {
            return 0;
        }
        if (true_soc_start<0||true_soc_end>Pi+1e-3) {
            return -1;
        }

        for (const auto& data : charge_table) {

            if (data.soc_start == static_cast<int>(std::round(true_soc_start)) &&
                data.soc_end == static_cast<int>(std::round(true_soc_end))) {
                return data.time_min;
            }
        }
        // cout<<"static_cast<int>(std::round(100*true_soc_start/Pi)):"<<static_cast<int>(std::round(100*true_soc_start/Pi))<<endl;
        // cout<<"static_cast<int>(std::round(100*true_soc_end/Pi)):"<<static_cast<int>(std::round(100*true_soc_end/Pi))<<endl;
        // cout<<"true_soc_start:"<<true_soc_start<<endl;
        // cout<<"true_soc_end:"<<true_soc_end<<endl;
        // cout<<"出错了"<<endl;
        // system("pause");
        // 没找到，返回-1或抛出异常
        return -1.0;
    }
    else
    {
        return (true_soc_end-true_soc_start)/eta;
    }


}
//????????????????????????
double Parameter::get_max_charging_energy(double true_soc_start, double true_soc_end, double max_charge) {
    if (true_soc_start == true_soc_end) {
        return 0.0;
    }
    if (true_soc_start < 0 || true_soc_end > Pi+1e-3) {
        std::cout << "Error: SOC out of range" << std::endl;
        return -1.0;
    }

    int start = static_cast<int>(std::round(true_soc_start));
    int end = static_cast<int>(std::round(true_soc_end));
    if (start ==  end) {
        return 0.0;
    }
    double max_end = std::round(true_soc_start + max_charge);

    double target_time = -1.0;
    // ???????????
    for (const auto& data : charge_table) {
        if (data.soc_start == start && data.soc_end == end) {
            target_time = data.time_min;
            break;
        }
    }

    if (target_time < 0) {
        std::cerr << "Error: No matching charge time found" << std::endl;
        return -1.0;
    }

    // ??? soc_end ????????????????? target_time
    int best_end = start;
    for (const auto& data : charge_table) {
        if (data.soc_start == start &&
            data.soc_end <= max_end &&
            std::abs(data.time_min - target_time) < 1e-6) {
            if (data.soc_end > best_end) {
                best_end = data.soc_end;
            }
            }
    }

    // ????????
    return best_end - true_soc_start;
}

double Parameter::get_best_soc_delta(double soc_start) {
    int soc_start_int = static_cast<int>(std::round(soc_start));
    std::vector<const ChargeData*> candidates;
    for (const auto& data : charge_table) {
        if (data.soc_start == soc_start_int && soc_start + data.soc_delta <= Pi) {
            candidates.push_back(&data);
        }
    }
    if (candidates.empty()) return 0.0;

    // ???????ε???Ч??
    double prev_eff = candidates[0]->soc_delta / candidates[0]->time_min;
    double best_soc_delta = candidates[0]->soc_delta;
    for (size_t i = 1; i < candidates.size(); ++i) {
        double delta_soc = candidates[i]->soc_delta - candidates[i-1]->soc_delta;
        double delta_time = candidates[i]->time_min - candidates[i-1]->time_min;
        if (delta_time <= 0) continue;
        double eff = delta_soc / delta_time;
        // ??????Ч?????????20%???????????
        if (eff < prev_eff * 0.7) {
            break;
        }
        prev_eff = eff;
        best_soc_delta = candidates[i]->soc_delta;
    }
    return best_soc_delta;
}


void Parameter::generate_distance_matrix() {
    // ??????????????????????????????????
    if (useCSV) {
        read_distance_from_csv();
    } else {

    }
}

//?????????????????????????????????
void Parameter::amplify_critical_distances() {
    // ????1?????????????ε???λ??ε???Σ?????????????
    for (int i = 1; i <= 5; i++) {        // ??ε??
        for (int j = 6; j <= 10; j++) {    // ??ε??
            // ??ε???ε????????10-20%
            distance[i][j] *= (1.0 + 0.15 * (sharedGenerator() % 100) / 100.0);
            distance[j][i] = distance[i][j]; // ????????
        }
    }

    // ????2????????????????????????
    double amplification_factor = 1.0;
    if (J >= 500) {
        amplification_factor = 1.1;  // ??????????????
    } else if (J >= 200) {
        amplification_factor = 1.15; // ?й???????е???
    } else {
        amplification_factor = 1.2;  // С??????????????
    }

    // ????3????????????????
    if (C == 1) {
        // ????????????????????????????????????????????
        for (int i = 1; i <= 10; i++) {
            for (int j = i + 1; j <= 10; j++) {
                if (distance[i][j] > 60.0) { // ??????????
                    distance[i][j] *= amplification_factor;
                    distance[j][i] = distance[i][j];
                }
            }
        }
    }

    if (isPrint) {
        std::cout << "??????????????????: " << amplification_factor << std::endl;
    }
}

// ??????????????????????????
void Parameter::analyze_distance_characteristics() {
    if (!isPrint) return;

    std::cout << "\n========== ??????????????? ==========" << std::endl;

    // ????????
    std::vector<double> all_distances;
    for (int i = 1; i <= 10; i++) {        // ????????????????
        for (int j = i + 1; j <= 10; j++) {
            all_distances.push_back(distance[i][j]);
        }
    }

    std::sort(all_distances.begin(), all_distances.end());

    double min_dist = all_distances.front();
    double max_dist = all_distances.back();
    double avg_dist = std::accumulate(all_distances.begin(), all_distances.end(), 0.0) / all_distances.size();

    std::cout << "??????????:" << std::endl;
    std::cout << "  ??С????: " << std::fixed << std::setprecision(2) << min_dist << " km" << std::endl;
    std::cout << "  ??????: " << std::fixed << std::setprecision(2) << max_dist << " km" << std::endl;
    std::cout << "  ???????: " << std::fixed << std::setprecision(2) << avg_dist << " km" << std::endl;

    // ??????????
    double max_energy_per_task = Delta * max_dist;
    double avg_energy_per_task = Delta * avg_dist;

    std::cout << "\n?????? (Delta=" << Delta << " kWh/km):" << std::endl;
    std::cout << "  ??????????: " << std::fixed << std::setprecision(2) << max_energy_per_task << " kWh" << std::endl;
    std::cout << "  ??????????: " << std::fixed << std::setprecision(2) << avg_energy_per_task << " kWh" << std::endl;
    std::cout << "  ?????????????: " << std::fixed << std::setprecision(1)
              << (max_energy_per_task / Pi) * 100 << "%" << std::endl;

    // ?????????
    int estimated_tasks_before_charging = static_cast<int>((Pi - gamma) / avg_energy_per_task);
    std::cout << "  ??????????????????: " << estimated_tasks_before_charging << " ??????" << std::endl;

    // ????????????
    std::cout << "\n???????????:" << std::endl;
    for (int c = 0; c < C; c++) {
        int cs_index = 11 + c;
        double max_dist_to_cs = 0.0;
        for (int i = 1; i <= 10; i++) {
            max_dist_to_cs = std::max(max_dist_to_cs, distance[i][cs_index]);
        }
        std::cout << "  CS" << (c+1) << " ???????: " << std::fixed << std::setprecision(2)
                  << max_dist_to_cs << " km" << std::endl;
    }

    std::cout << "=====================================" << std::endl;
}

// ???????????
void Parameter::save_tasks_to_file(const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "??????????????" << std::endl;
        return;
    }

    // д????
    outfile << "??????,????????,???,???,????,??????,???????" << std::endl;

    // д??????????
    for (const auto& task_pair : tasks) {
        const auto& task = task_pair.second; // ??? second ???? Task ????
        outfile << task.no << ",";
        outfile << (task.type == Task::UNLOADING ? "ж??" : "???") << ",";

        // ???????????????λ??????
        if (task.startLoc == 0) {
            outfile << "O";
        } else if (task.startLoc >= 1 && task.startLoc <= 5) {
            outfile << "QC" << task.startLoc;
        } else if (task.startLoc >= 6 && task.startLoc <= 15) {
            outfile << "YC" << (task.startLoc - 5);
        } else {
            outfile << "CS" << (task.startLoc - 15);
        }

        outfile << ",";

        if (task.endLoc == 0) {
            outfile << "O";
        } else if (task.endLoc >= 1 && task.endLoc <= 5) {
            outfile << "QC" << task.endLoc;
        } else if (task.endLoc >= 6 && task.endLoc <= 15) {
            outfile << "YC" << (task.endLoc - 5);
        } else {
            outfile << "CS" << (task.endLoc - 15);
        }

        outfile << "," << task.distance
                << "," << task.tl
                << std::endl;
    }

    outfile.close();
    std::cout << "????????浽 " << filename << std::endl;
}

// ??????????AGVs
void Parameter::assign_tasks_to_trucks() {
    // ????????AGV???????б?
    task_sequences.resize(V);

    // ??????????????????
    int assigned_tasks = 0;

    // ????J_v????????????AGV
    for (int v = 0; v < V; v++) {
        int tasks_for_agv = J_v[v];

        for (int j = 0; j < tasks_for_agv && assigned_tasks < tasks.size(); j++) {
            task_sequences[v].push_back(&tasks[assigned_tasks]);
            assigned_tasks++;
        }
    }

    // ????????????????
    if (assigned_tasks != tasks.size()) {
        std::cerr << "????: ???????????! ????? " << assigned_tasks
                  << " ????????? " << tasks.size() << " ??????" << std::endl;
    }
}

void Parameter::generate_tasks() {
    tasks.clear();

    int task_no = 0;
    int num_QC = 5;  // QC1-QC5
    int num_YC = 10; // YC1-YC10

    // ????????????J????????
    for (int j = 0; j < J; j++) {
        // ?????????????????????ж????
        std::uniform_int_distribution<int> type_dist(0, 1);
        bool is_loading = type_dist(sharedGenerator) == 1; // 1??????????0???ж??????

        Task::TaskType task_type = is_loading ? Task::LOADING : Task::UNLOADING;

        // ?????????QC??YC
        std::uniform_int_distribution<int> qc_dist(1, num_QC);
        int qc_idx = qc_dist(sharedGenerator);

        std::uniform_int_distribution<int> yc_dist(1, num_YC);
        int yc_idx = yc_dist(sharedGenerator);

        // ?????distance?????е?????
        int qc_loc = qc_idx;  // QC????????1???
        int yc_loc = num_QC + yc_idx;  // YC????????(num_QC+1)???

        // ???????????????????????
        int start_loc, end_loc;

        if (task_type == Task::UNLOADING) {
            // ж???????QC??YC
            start_loc = qc_loc;
            end_loc = yc_loc;
        } else {
            // ????????YC??QC
            start_loc = yc_loc;
            end_loc = qc_loc;
        }

        // ??distance??????????
        double task_distance = distance[start_loc][end_loc]/1e3;
        //cout<<"task_distance:"<<task_distance<<"start_loc:"<<start_loc<<"end_loc:"<<end_loc<<"distance[start_loc][end_loc]:"<<distance[start_loc][end_loc]<<endl;
        // ????????
        //double travel_time = (task_distance / V_loading) * 1.5 + 10; // 1.5?????
        double travel_time = (task_distance / V_loading) * 2.7   ; // 1.8???????λ?????

        //cout<<"task_no:"<<task_no<<"travel_time:"<<travel_time<<"task_distance:"<<task_distance<<endl;

        tasks[task_no] = Task(task_no, start_loc, end_loc, task_distance, travel_time, task_type);
        task_no++;
    }
    //system("pause");
}

void Parameter::print_tasks() {
    std::cout << "Generated tasks:\n";
    std::cout << std::left
              << std::setw(5) << "No"
              << std::setw(8) << "Start"
              << std::setw(8) << "End"
              << std::setw(12) << "Distance"
              << std::setw(12) << "LoadTime"
              << std::setw(12) << "UnloadTime"
              << std::setw(12) << "FromCS_Time"
              << std::setw(12) << "ToCS_Time"
              << std::endl;

    std::cout << std::string(142, '-') << std::endl;

    for (const auto& task_pair : tasks) {
        const auto& task = task_pair.second;

        std::cout << std::left
                  << std::setw(5) << task.no
                  << std::setw(8) << task.startLoc
                  << std::setw(8) << task.endLoc
                  << std::setw(12) << std::fixed << std::setprecision(4) << task.distance
                  << std::setw(12) << std::fixed << std::setprecision(4) << task.tl
                  << std::setw(12) << std::fixed << std::setprecision(4) << task.te
                  << std::setw(12) << std::fixed << std::setprecision(4) << task.tr[0]
                  << std::setw(12) << std::fixed << std::setprecision(4) << task.tf[0]
                  << std::endl;
    }

    std::cout << "\nTask assignments for each AGV:\n";
    for (int v = 0; v < V; v++) {
        std::cout << "AGV #" << v << " assigned " << task_sequences[v].size() << " tasks: ";
        if (!task_sequences[v].empty()) {
            std::cout << task_sequences[v][0]->no << " - ";
            std::cout << task_sequences[v][task_sequences[v].size()-1]->no << " ";
        }
        std::cout << std::endl;
    }

    // ???????????????????
    std::cout << "\nCharging Station Information:\n";
    std::cout << "Number of charging stations: 1" << std::endl;

    // ??????AGV????????????
    std::cout << "\nAGV Charging Station Access Summary:\n";
    std::cout << std::left
              << std::setw(5) << "AGV"
              << std::setw(12) << "Tasks"
              << std::setw(15) << "Avg_ToCS_Time"
              << std::setw(15) << "Avg_FromCS_Time"
              << std::setw(15) << "Avg_ToCS_Energy"
              << std::setw(15) << "Avg_FromCS_Energy"
              << std::endl;

    std::cout << std::string(77, '-') << std::endl;

    for (int v = 0; v < V; v++) {
        if (task_sequences[v].empty()) continue;

        double total_te = 0.0, total_tb = 0.0, total_ce = 0.0, total_cb = 0.0;
        int count = task_sequences[v].size();

        std::cout << std::left
                  << std::setw(5) << v
                  << std::setw(12) << count
                  << std::setw(15) << std::fixed << std::setprecision(4) << (total_te / count)
                  << std::setw(15) << std::fixed << std::setprecision(4) << (total_tb / count)
                  << std::setw(15) << std::fixed << std::setprecision(4) << (total_ce / count)
                  << std::setw(15) << std::fixed << std::setprecision(4) << (total_cb / count)
                  << std::endl;
    }
}


void Parameter::generate_travel_time_scenarios() {
    // ??????????
    time_tilde.clear();

    // ??????????????????????
    double total_task_time = 0.0;
    int task_count = 0;

    for (const auto& task_pair : this->tasks) {
        total_task_time += task_pair.second.tl;
        task_count++;
    }
    double avg_task_time = (task_count > 0) ? (total_task_time / task_count) : 0.0;

    // ??????????????????
    double high_variance_mean = avg_task_time * 0.4;  // ?????????????????????????40%??
    double low_variance_mean = avg_task_time * 0.1;   // ?????????????????????????10%??

    double high_variance_std = high_variance_mean * 0.2;  // ?????????????30%
    double low_variance_std = low_variance_mean * 0.2;    // ?????????????30%

    std::normal_distribution<double> high_dist(high_variance_mean, high_variance_std);
    std::normal_distribution<double> low_dist(low_variance_mean, low_variance_std);

    int high_samples = omega_num * 3 / 5;  // 60%??????

    // ?????????????????
    while (time_tilde.size() < high_samples) {
        double value = high_dist(sharedGenerator);
        if (value >= 0) {  // ????????????????
            time_tilde.push_back(value);
        }
    }

    // ?????????????????
    while (time_tilde.size() < omega_num) {
        double value = low_dist(sharedGenerator);
        if (value >= 0) {  // ????????????????
            time_tilde.push_back(value);
        }
    }

    // ???????????????????
    std::sort(time_tilde.begin(), time_tilde.end());

//    std::cout << "Generated " << time_tilde.size() << " energy uncertainty scenarios." << std::endl;
//    std::cout << "Average task energy: " << avg_task_time << std::endl;
//    std::cout << "Min energy uncertainty: " << time_tilde.front() << std::endl;
//    std::cout << "Max energy uncertainty: " << time_tilde.back() << std::endl;

}


// ????Wasserstein???????????
void Parameter::calculate_time_dro() {
    xi_ave_saa = calculate_time_ave_saa();
    xi_ave_dro = calculate_time_ave_dro();
    xi_cvar_dro = calculate_time_cvar_dro();
}

// ????Wasserstein???????????
double Parameter::calculate_time_ave_saa() {
    // ??????????в?????????????
    double sum = std::accumulate(time_tilde.begin(), time_tilde.end(), 0.0);
    // ????????Theta?????????Wasserstein??????????
    return sum / omega_num ;
}

// ????Wasserstein???????????
double Parameter::calculate_time_ave_dro() {
    return this->xi_ave_saa + Theta ;
}

// ????Wasserstein???????CVaR
double Parameter::calculate_time_cvar_dro() {
    // ?????????????????
    std::vector<double> sorted_tilde = time_tilde;

    // ??????С?????????
    std::sort(sorted_tilde.begin(), sorted_tilde.end(), std::greater<double>());

    // ????CVaR?????????????minIndex??????????
    double sum = 0.0;
    for (int i = 0; i < minIndex; i++) {
        sum += sorted_tilde[i];
    }

    // ???threshold??????????CVaR
    double cvar = sum / threshold;

    // ???threshold?????????????CVaR????????
    if (minIndex != maxIndex) {
        cvar += (1 - minIndex / threshold) * sorted_tilde[minIndex];
    }

    // ???Wasserstein??
    double wasserstein_term = Theta / epsilon;
    double time_cvar = cvar + wasserstein_term;

    return time_cvar;
}

void Parameter::generate_agv_init_SOC() {
    // ???task_sequences??????????
    if (task_sequences.empty()) {
        std::cerr << "????: ?????????assign_tasks_to_agvs????!" << std::endl;
        return;
    }

    varphies.clear();
    for (int v = 0; v < V; v++) {
        // ???AGV??б????????????????????
        if (task_sequences[v].empty()) {
            normal_distribution<double> dis = normal_distribution<double>(0.3, 0.05);
            double var;
            do {
                var = dis(sharedGenerator);
            } while (var < this->gamma);

            varphies.push_back(var);
            continue;
        }

        //double min_required_varphi = gamma + (delta * task_sequences[v][0]->te + Delta * task_sequences[v][0]->tl) /this->Pi;
        double min_required_varphi = gamma + (task_sequences[v][0]->ce) /this->Pi;


        // ????????Щ????????????10%????????
        //min_required_varphi *= 1.5;
        min_required_varphi *= 2.0;

        // ??????????????????????С?????????С?
        normal_distribution<double> dis = normal_distribution<double>(min_required_varphi + 0.1, 0.05);
        double var;
        do {
            var = dis(sharedGenerator);
        } while (var < min_required_varphi);
    // cout<<"var:"<<var<<endl;
    // system("pause");
        varphies.push_back(var);

    }
}

// void Parameter::generate_all_task() {
//     // ?????????????г???λ??????
//     int CS_start = 1 + 5 + 10;// ?????????г?????????7?????????7,8,9,10???4????????
//
//     // ???????????
//     for (int v = 0; v < V; v++) {
//         std::vector<Task*>& truck_tasks = this->task_sequences[v];
//         std::vector<double> task_start_times(truck_tasks.size(), 0.0);
//         std::vector<double> task_end_times(truck_tasks.size(), 0.0);
//         vector<double> temp_tf(C,0);
//         vector<double> temp_cf(C,0);
//         // ?????????????????
//         for (int j = 0; j < truck_tasks.size(); j++) {
//             // ??????????
//             Task* current_task = truck_tasks[j];
//
//             int prev_task_end_loc, task_start_loc, task_end_loc;
//
//             // ??????????????????
//             current_task->initialize_cs_times(C);
//
//             // ????????????
//             if (j == 0) {
//                 prev_task_end_loc = 0;  // ???????λ????????????????0??
//             } else {
//                 // ??????????????????????????????λ?ó???
//                 prev_task_end_loc = truck_tasks[j - 1]->endLoc;
//             }
//             task_start_loc = current_task->startLoc;
//             task_end_loc = current_task->endLoc;
//
//             // ????????λ???????????????????????????
//             current_task->te = distance[prev_task_end_loc][task_start_loc] / V_unloading;
//             // ??????????????????????????
//             current_task->ce = delta * distance[prev_task_end_loc][task_start_loc];  // c^u_{v,j} = delta * distance
//             // std::cout << "prev=" << prev_task_end_loc << ", start=" << task_start_loc
//             //           << ", distance=" << distance[prev_task_end_loc][task_start_loc]
//             //           << ", V_unloading=" << V_unloading
//             //           << ", t_u=" << current_task->tu << std::endl;
//             // ???????????????????
//             if (j == 0) {
//                 task_start_times[j] = current_task->te;  // ?????????????????????????????
//             } else {
//                 task_start_times[j] = task_end_times[j-1] + current_task->te;  // ????????????????????????
//             }
//
//             // ??????????????????
//             // ??????????????????????????????
//             current_task->cl = Delta * current_task->distance;  // c^l_{v,j} = Delta * task_distance
//
//             // ??????????????
//             task_end_times[j] = task_start_times[j] + current_task->tl;
//
//             // ???????????????R??????????????????Щ???????????
//             // ????????????????????????????е???????
//             std::uniform_real_distribution<double> dist(1.2, 1.4);  // ???1.20??1.40??????????
//             double uncertainty_factor = dist(sharedGenerator);
//             double ready_time = task_start_times[j] * uncertainty_factor;
//
//             // **?????????????????????????????????????????**
//             for (int c = 0; c < C; c++) {
//                 int cs_loc = CS_start + c;  // ?????????г?????????7, 8, 9, 10??
//
//                 // ?????飺????????????????????????Χ
//                 if (cs_loc >= distance.size()) {
//                     std::cerr << "??????????? " << cs_loc << " ???????????Χ" << std::endl;
//                     continue;
//                 }
//
//                 // ????????????????c?????????
//                 double time_to_cs = distance[task_end_loc][cs_loc] / V_unloading;
//                 current_task->tr[c] = time_to_cs;
//                 // ??????????????????????????
//                 current_task->cr[c] = delta * distance[task_end_loc][cs_loc];  // c^e_{v,j,c} = delta * distance
//
//                 // **????????????c??????????????????????????????????**
//                 if (j + 1 < truck_tasks.size()) {
//                     int next_task_start_loc = truck_tasks[j + 1]->startLoc;
//
//                     double time_from_cs = distance[cs_loc][next_task_start_loc] / V_unloading;
//
//                     //current_task->tb[c] = time_from_cs;
//                     //current_task->cb[c] = delta * distance[cs_loc][next_task_start_loc];  // c^b_{v,j,c} = delta * distance
//                     if (j==0) {
//                         current_task->tf[c] =0;
//                         current_task->cf[c] =0;
//                         temp_tf[c] = time_from_cs;
//                         temp_cf[c] = delta * distance[cs_loc][next_task_start_loc];
//                     }
//                     else {
//                         current_task->tf[c] = temp_tf[c];
//                         temp_tf[c] = time_from_cs;
//                         current_task->cf[c] = temp_cf[c];
//                         temp_cf[c] = delta * distance[cs_loc][next_task_start_loc];  // c^b_{v,j,c} = delta * distance
//                     }
//
//
//                     // ??????????????????????????
//                     //current_task->cb[c] = delta * distance[cs_loc][next_task_start_loc];  // c^b_{v,j,c} = delta * distance
//                     //next_task->cb[c] = delta * distance[cs_loc][next_task_start_loc];  // c^b_{v,j,c} = delta * distance
//                 } else {
//                     // // ????????????????????????????0??
//                     // double time_from_cs = distance[cs_loc][0] / V_unloading;
//                     // current_task->tb[c] = time_from_cs;
//                     // // ??????????????????????????
//                     // current_task->cb[c] = delta * distance[cs_loc][0];  // c^b_{v,j,c} = delta * distance
//                     current_task->tf[c] = temp_tf[c];
//                     temp_tf[c] = distance[cs_loc][0] / V_unloading;
//                     current_task->cf[c] = temp_cf[c];
//                     temp_cf[c] = delta * delta * distance[cs_loc][0];
//
//
//                 }
//             }
//
//
//             // ?????????????????????????????
//         }
//         //system("pause");
//     }
//
//
// }
// ??????????????????????????????
void Parameter::generate_all_task() {
    // ????????????????
    int CS_start = 1 + 5 + 10; // ??? + QC???? + YC????????CS1??????

    // ???????AGV
    for (int v = 0; v < V; v++) {
        std::vector<Task*>& agv_tasks = this->task_sequences[v];
        std::vector<double> task_start_times(agv_tasks.size(), 0.0);
        std::vector<double> task_end_times(agv_tasks.size(), 0.0);

        // ????AGV?????????
        for (int j = 0; j < agv_tasks.size(); j++) {
            // ??????????
            Task* current_task = agv_tasks[j];
            int from_node, to_node;

            // ?????????????????
            current_task->initialize_cs_times(C);

            // ????????????
            if (j == 0) {
                from_node = 0;  // AGV???λ???0
            } else {
                // ??????????????????????????????????????????
                from_node = agv_tasks[j-1]->endLoc;
            }
            to_node = current_task->startLoc;

            // ????????λ???????????????????????
            current_task->te = (distance[from_node][to_node]/1e3) / V_unloading;
            // ??????????????????????????
            //current_task->ce = delta * current_task->te;  // ??????????
            current_task->ce = delta * (distance[from_node][to_node]/1e3);  // ??????????

            std::uniform_real_distribution<double> dist(2, 5);  // ???1.00??1.30???????????
            double uncertainty_factor = dist(sharedGenerator);
            double ready_time = 0.0;

            if (j == 0) {
                task_start_times[j] = current_task->te;  // ??????????????λ??????????
                ready_time = task_start_times[j] * uncertainty_factor;
            } else {
                task_start_times[j] = task_end_times[j-1] + current_task->te;  // ????????????????????????
                ready_time = task_end_times[j-1] + current_task->te * uncertainty_factor;
            }

            current_task->ready_time = ready_time;
            task_end_times[j] = ready_time + current_task->tl;
            current_task->cl = Delta * current_task->distance;

            // ?????????????????????????????????(tr)
            for (int c = 0; c < C; c++) {
                int cs_loc = CS_start + c;  // ?????????????е?????
                double time_to_cs = (distance[current_task->endLoc][cs_loc]/1e3) / V_unloading;
                current_task->tr[c] = time_to_cs;
                //??????
                //current_task->cr[c] = delta * time_to_cs;
                current_task->cr[c] = delta * (distance[current_task->endLoc][cs_loc]/1e3);
                if (j + 1 < agv_tasks.size()) {
                    int next_task_start_loc = agv_tasks[j + 1]->startLoc;
                    double time_from_cs = (distance[cs_loc][next_task_start_loc]/1e3) / V_unloading;
                    current_task->tf[c] = time_from_cs;
                    //???????????
                    //current_task->cf[c] = delta * time_from_cs;
                    current_task->cf[c] = delta * (distance[cs_loc][next_task_start_loc]/1e3);
                }
                else {
                    //????????1
                    //int next_task_start_loc = 0;
                    int next_task_start_loc = 1;
                    double time_from_cs = (distance[cs_loc][next_task_start_loc]/1e3) / V_unloading;

                    current_task->tf[c] = time_from_cs;
                    //???????????
                    //current_task->cf[c] = delta * time_from_cs;
                    current_task->cf[c] = delta *(distance[cs_loc][next_task_start_loc]/1e3);

                }


            }


        }
    }
}

// C++
void Parameter::reset() {
    // distance.clear();
    // cs_schedule.clear();
    // cs_available_time.clear();
    // J_v.clear();
    // varphies.clear();
    // time_tilde.clear();
    // zeta_tilde.clear();
    // fileName.clear();
    // tasks.clear();
    // task_sequences.clear();
    //
    // V = 0;
    // J = 0;
    // C = 0;
    // V_loading = 60;
    // V_unloading = 60;
    // Delta = 0.93;
    // delta = 0.75;
    // tu = 0.5;
    // Pi = 225;
    // gamma_percent = 0.20;
    // gamma = Pi * gamma_percent;
    // omega_num = 200;
    // epsilon = 0.2;
    // Theta = 0.1;
    // wassterTemp = 0;
    // M = 1000;
    // time_limit = 3600;
    // xi_ave_saa = 0;
    // xi_ave_dro = 0;
    // xi_cvar_dro = 0;
    // threshold = 0;
    // maxIndex = 0;
    // minIndex = 0;
    // zeta_ave_saa = 0;
    // zeta_ave_dro = 0;
    // zeta_cvar_dro = 0;
    // isPrint = true;
    // useCSV = true;
    // keepAllBounds = true;
    // max_time = 3600;
    // tolerance = 0.0000000001;
    // max_iterations = 10e6;
}

// ??Parameter.cpp?????????
void Parameter::generate_energy_uncertainty_scenarios() {
    // ??????????????
    zeta_tilde.clear();

    // ???????????????????
    double total_energy_consumption = 0.0;
    int task_count = 0;

    for (const auto& task_pair : this->tasks) {
        // ????????????????????????????????????????
        double task_energy = Delta * task_pair.second.distance;  // ????????????????
        total_energy_consumption += task_energy;
        task_count++;
    }

    double avg_task_energy = (task_count > 0) ? (total_energy_consumption / task_count) : 0.0;

    // ??????????????????
    double high_variance_mean = avg_task_energy * 0.4;  // ?????????????????????????40%??
    double low_variance_mean = avg_task_energy * 0.1;   // ?????????????????????????10%??

    double high_variance_std = high_variance_mean * 0.2;  // ?????????????20%
    double low_variance_std = low_variance_mean * 0.2;    // ?????????????20%

    std::normal_distribution<double> high_dist(high_variance_mean, high_variance_std);
    std::normal_distribution<double> low_dist(low_variance_mean, low_variance_std);

    int high_samples = omega_num * 3 / 5;  // 60%??????

    // ?????????????????
    while (zeta_tilde.size() < high_samples) {
        double value = high_dist(sharedGenerator);
        if (value >= 0) {  // ????????????????
            zeta_tilde.push_back(value);
        }
    }

    // ?????????????????
    while (zeta_tilde.size() < omega_num) {
        double value = low_dist(sharedGenerator);
        if (value >= 0) {  // ????????????????
            zeta_tilde.push_back(value);
        }
    }

    // ???????????????????
    std::sort(zeta_tilde.begin(), zeta_tilde.end());

//     ????????????????????????????
//     std::cout << "Generated " << zeta_tilde.size() << " energy uncertainty scenarios." << std::endl;
//     std::cout << "Average task energy: " << avg_task_energy << std::endl;
//     std::cout << "Min energy uncertainty: " << zeta_tilde.front() << std::endl;
//     std::cout << "Max energy uncertainty: " << zeta_tilde.back() << std::endl;
}

double Parameter::calculate_zeta_ave_saa() {
    // ??????????????????????????
    double sum = std::accumulate(zeta_tilde.begin(), zeta_tilde.end(), 0.0);
    return sum / omega_num;
}

double Parameter::calculate_zeta_ave_dro() {
    // Wasserstein???????????? = SAA????? + Theta????
    return this->zeta_ave_saa + Theta;
}

double Parameter::calculate_zeta_cvar_dro() {
    // ?????????????????????
    std::vector<double> sorted_zeta = zeta_tilde;

    // ??????С?????????
    std::sort(sorted_zeta.begin(), sorted_zeta.end(), std::greater<double>());

    // ????CVaR?????????????minIndex??????????
    double sum = 0.0;
    for (int i = 0; i < minIndex; i++) {
        sum += sorted_zeta[i];
    }

    // ???threshold??????????CVaR
    double cvar = sum / threshold;

    // ???threshold?????????????CVaR????????
    if (minIndex != maxIndex) {
        cvar += (1 - minIndex / threshold) * sorted_zeta[minIndex];
    }

    // ???Wasserstein??
    double wasserstein_term = Theta / epsilon;
    double zeta_cvar = cvar + wasserstein_term;

    return zeta_cvar;
}

void Parameter::calculate_zeta_dro() {
    zeta_ave_saa = calculate_zeta_ave_saa();
    zeta_ave_dro = calculate_zeta_ave_dro();
    zeta_cvar_dro = calculate_zeta_cvar_dro();
}



void Parameter::process_precision() {
    adjustFloatPrecision(this->epsilon);
    adjustFloatPrecision(this->Theta);
    this->wassterTemp = this->Theta / this->epsilon;
    this->threshold = this->epsilon * this->omega_num;
    this->maxIndex = ceil(this->threshold);
    this->minIndex = floor(this->threshold);

}

// void Parameter::adjustFloatPrecision(double &value) {
//     // ?????С????????????????????????
//     value = std::round(value * 100000.0) / 100000.0;  // ??????????λС??
// }

void Parameter::adjustFloatPrecision(double &value) {
    // ?????С????????????????????????
    value = std::round(value * 1000.0) / 1000.0;  // ??????????λС??
}