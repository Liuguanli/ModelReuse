#ifndef PGM_
#define PGM_

#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "pgm_index.hpp"
#include "pgm_index_dynamic.hpp"

template<int error>
class PGMX
{
public:
    DynamicPGMIndex<uint64_t, uint64_t, PGMIndex<uint64_t, error>> pgm_;

    void build(const std::vector<uint64_t> &data)
    {
        std::vector<std::pair<uint64_t, uint64_t>> reformatted_data;
        reformatted_data.reserve(data.size());
        uint64_t length = data.size();
        for (uint64_t i = 0; i < length; i++)
        {
            reformatted_data.emplace_back(data[i], i);
        }
        auto start = std::chrono::high_resolution_clock::now();
        pgm_ = decltype(pgm_)(reformatted_data.begin(), reformatted_data.end());
        auto end = std::chrono::high_resolution_clock::now();
        std::cout<< "build time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;
    }

    void find(const uint64_t lookup_key)
    {
        auto index = pgm_.find(lookup_key);
    }

    void insert(const uint64_t insert_key, const uint64_t index)
    {
        pgm_.insert(insert_key, index);
    }

};

#endif