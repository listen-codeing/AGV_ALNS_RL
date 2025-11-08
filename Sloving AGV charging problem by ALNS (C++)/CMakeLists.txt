cmake_minimum_required(VERSION 3.22)

project(predict_agv_charging C)
# ???? C++17 ???
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CXX "enable C++ compilation" ON)

if(CXX)
    enable_language(CXX)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Visual Studio compiler with static runtime libraries
if(MSVC AND MT)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
endif()

find_package(GUROBI REQUIRED)

include_directories(${GUROBI_INCLUDE_DIRS})

# list source files here
set(sources main.cpp
        TestUnit.cpp
        TestUnit.h
        random_seed.h
        main.cpp
        Model.cpp
        Model.h
        Model.h
        Parameter.cpp
        Parameter.h
        Parameter.h
        Heuristic/ADestroyOperator.h
        Heuristic/ALNS.cpp
        Heuristic/ALNS.h
        Heuristic/ALNS_Iteration_Status.h
        Heuristic/ALNS_Parameters.cpp
        Heuristic/ALNS_Parameters.h

        Heuristic/AOperator.h
        Heuristic/AOperatorManager.h
        Heuristic/ARepairOperator.h
        Heuristic/CoolingSchedule_Parameters.cpp
        Heuristic/CoolingSchedule_Parameters.h
        Heuristic/CoolingScheduleFactory.cpp
        Heuristic/CoolingScheduleFactory.h
        Heuristic/ExponentialCoolingSchedule.cpp
        Heuristic/ExponentialCoolingSchedule.h
        Heuristic/IAcceptanceModule.h
        Heuristic/IBestSolutionManager.h
        Heuristic/ICoolingSchedule.h


        Heuristic/ISolution.h
        Heuristic/LinearCoolingSchedule.cpp
        Heuristic/LinearCoolingSchedule.h
        Heuristic/OperatorManager.cpp
        Heuristic/OperatorManager.h
        Heuristic/SimpleBestSolutionManager.cpp
        Heuristic/SimpleBestSolutionManager.h
        Heuristic/SimulatedAnnealing.cpp
        Heuristic/SimulatedAnnealing.h
        Heuristic/Statistics.cpp
        Heuristic/Statistics.h
        Heuristic/Charging_Critical_Removal.cpp
        Heuristic/Charging_Critical_Removal.h
        Heuristic/Charging_Random_Removal.cpp
        Heuristic/Charging_Random_Removal.h

        Heuristic/AGV_solution.cpp
        Heuristic/AGV_solution.h
        Heuristic/AGV_solution.h
        Heuristic/ILocalSearch.h
        Heuristic/ILocalSearchManager.h
        Heuristic/SimpleLocalSearchManager.h
        Heuristic/SimpleLocalSearchManager.cpp
        Heuristic/ALocalSearchOperator.h


        AGV.cpp
        AGV.h
        Task.cpp
        Task.h
        GurobiModel.cpp
        GurobiModel.h
        GurobiModel.h
        Heuristic/Random_Best_Repair.cpp
        Heuristic/Random_Best_Repair.h
        Heuristic/Random_Random_Repair.cpp
        Heuristic/Random_Random_Repair.h
        Heuristic/Random_Time_Repair.cpp
        Heuristic/Random_Time_Repair.h
        Heuristic/Greedy_Random_Repair.cpp
        Heuristic/Greedy_Random_Repair.h
        Heuristic/Adaptive_Random_Repair.cpp
        Heuristic/Adaptive_Random_Repair.h
        Heuristic/Charging_Worst_Removal.cpp
        Heuristic/Charging_Worst_Removal.h
        Heuristic/Greedy_Best_Repair.cpp
        Heuristic/Greedy_Best_Repair.h
        Heuristic/Adaptive_Best_Repair.cpp
        Heuristic/Adaptive_Best_Repair.h
        Heuristic/Adaptive_Time_Repair.cpp
        Heuristic/Adaptive_Time_Repair.h
        Heuristic/Greedy_Time_Repair.cpp
        Heuristic/Greedy_Time_Repair.h
        Heuristic/NeighSearch_Random_Ls.cpp
        Heuristic/NeighSearch_Random_Ls.h
        Heuristic/Station_Random_Removal.cpp
        Heuristic/Station_Random_Removal.h
        Heuristic/Station_Critical_Removal.cpp
        Heuristic/Station_Critical_Removal.h
        Heuristic/Station_Worst_Removal.cpp
        Heuristic/Station_Worst_Removal.h
)

# include directories containing header files
include_directories(Function.h)
include_directories(Node.h)

add_executable(${CMAKE_PROJECT_NAME} ${sources})

if(CXX)
    set(CMAKE_CXX_STANDARD 17)
    add_compile_options(/std:c++latest)
    target_link_libraries(${CMAKE_PROJECT_NAME} optimized ${GUROBI_CXX_LIBRARY}
            debug ${GUROBI_CXX_DEBUG_LIBRARY})
endif()

target_link_libraries(${CMAKE_PROJECT_NAME} ${GUROBI_LIBRARY})


if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    include(FeatureSummary)
    feature_summary(WHAT ALL)
endif()

# openMP ????
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()