#ifndef B_TREE
#define B_TREE

#include "../../../utils/tracking_allocator.h"
#include <stx/btree_multimap.h>
template<int n>
class Btree
{
public:
    uint64_t total_allocation_size = 0;
    uint64_t data_size_ = 0;

    Btree()
    {}

    template<int fanout>
    struct traits : stx::btree_default_map_traits<uint64_t, uint64_t>
    {
        static const int leafslots = fanout;
        static const int innerslots = fanout;
    };

    stx::btree_multimap<uint64_t,
                      uint64_t,
                      std::less<uint64_t>,
                      traits<n>> btree_;

    void build(const std::vector<uint64_t>& data)
    {
        std::vector<std::pair<uint64_t, uint64_t>> reformatted_data;
        reformatted_data.reserve(data.size());
        uint64_t length = data.size();
        for (uint64_t i = 0; i < length; i++)
        {
            reformatted_data.emplace_back(data[i], i);
        }
        auto start = std::chrono::high_resolution_clock::now();
        btree_.bulk_load(reformatted_data.begin(), reformatted_data.end());
        auto end = std::chrono::high_resolution_clock::now();
        std::cout<< "build time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;

    }

    void find(const uint64_t lookup_key)
    {
        auto it = btree_.find(lookup_key);
        uint64_t guess = it->second;
        // std::cout<< "guess: " << guess << " lookup_key: " << lookup_key << " " << "result: " << it->first << std::endl;
    }

    void insert(const uint64_t insert_key, const uint64_t index)
    {
        btree_.insert(insert_key, index);
    }

};

#endif