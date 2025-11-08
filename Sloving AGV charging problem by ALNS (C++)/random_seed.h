//
// Created by limin on 2023/11/4.
//

#ifndef QC_AGV_RANDOM_SEED_H
#define QC_AGV_RANDOM_SEED_H

#include <random>
#include <cstdlib>

static std::default_random_engine sharedGenerator(1);
static std::default_random_engine another_sharedGenerator(-1);

static unsigned int randSeed = 456; // ???????????

#endif //QC_AGV_RANDOM_SEED_H
