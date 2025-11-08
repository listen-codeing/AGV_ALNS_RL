

#include "TestUnit.h"
#include "random_seed.h"

int main(int argc, const char * argv[]) {

    vector<TestUnit> testModels;
    //HEURISTICS
    TestUnit test_exact_DRO2("small", SolveMethod::HEURISTICS, true, ModelType::DETERMINISTIC, true);
    testModels.push_back(test_exact_DRO2);
    for (auto &testModel : testModels){
        testModel.run();
    }
    return 0;
}



