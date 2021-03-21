#include <iostream>
#include "include/RMI_NN.h"
#include "include/RMRT.h"
#include "include/RMRT_LR.h"
#include "include/Btree.h"
#include "include/PGM.h"
#include "utils/SynData.h"
#include "utils/ExpRecorder.h"
#include "utils/FileReader.h"
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <torch/torch.h>

// #include <nlohmann/json.hpp>
// using json = nlohmann::json;

// #include <dirent.h>

void exe_rmi(std::vector<uint64_t> entities, uint64_t branch, string tag)
{
    RMI_NN rmi(true, tag, branch);
    ExpRecorder exp_recorder;
    if (branch == 1)
    {
        std::cout << "build_one_layer" << std::endl;
        rmi.build_one_layer(exp_recorder, entities);
    }
    else
    {
        std::cout << "build_two_layer" << std::endl;
        rmi.build_two_layer(exp_recorder, entities);
    }
    std::cout << "build time: " << exp_recorder.build_time << std::endl;
    std::cout << "model size: " << rmi.get_size() << std::endl;
    uint64_t N = entities.size();

    uint64_t insert_time = 0;
    std::cout << "N: " << N << std::endl;
    rmi.print_num();
    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        rmi.insert_two_layer_2(exp_recorder, entities[i] + 1, i + 1);
        auto end2 = std::chrono::high_resolution_clock::now();
        uint64_t temp = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        insert_time += temp;
        // std::cout << "insert time-----------: " << temp << std::endl;
        if ((i + 1) % (N / 100) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 100)) << "%: " << insert_time / i << std::endl;
        }
    }
    std::cout << "exp_recorder.prediction_time: " << exp_recorder.prediction_time / N << std::endl;
    std::cout << "exp_recorder.insert_time_inner: " << exp_recorder.insert_time_inner / N << std::endl;
    std::cout << "exp_recorder.insert_time: " << exp_recorder.insert_time / N << std::endl;
    std::cout << "exp_recorder.rebuild_time: " << exp_recorder.rebuild_time / exp_recorder.rebuild_num << std::endl;
    std::cout << "exp_recorder.rebuild_num: " << exp_recorder.rebuild_num << std::endl;
    std::cout << "insert time1: " << insert_time / N << std::endl;
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time2: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;
    auto start2 = std::chrono::high_resolution_clock::now();
    // rmi.print_num();
    int num = 0;
    exp_recorder.clear();
    for (size_t i = 0; i < N; i++)
    {
        rmi.search_after_insertion(exp_recorder, entities[i] + 1);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "exp_recorder.vector_visit_time: " << exp_recorder.vector_visit_time / N << std::endl;
    std::cout << "exp_recorder.prediction_time: " << exp_recorder.prediction_time / N << std::endl;
    std::cout << "exp_recorder.time: " << exp_recorder.time / N << std::endl;
    std::cout << "search time2: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
    exp_recorder.clear();
}

void exe_rmrt(std::vector<uint64_t> entities, uint64_t page_size, uint64_t threshold, string tag)
{
    uint64_t N = entities.size();
    RMRT rmrt(page_size, threshold, tag, 0);
    rmrt.is_root = true;
    rmrt.N = N;
    auto start = std::chrono::high_resolution_clock::now();
    // rmrt.build_recursively(entities.begin(), entities.size(), "" , 0);
    rmrt.build_recursively(entities, "", 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "build time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;
    std::cout << "size: " << rmrt.get_size() << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    uint64_t insert_time = 0;
    for (size_t i = 0; i < N; i++)
    {
        // std::cout << "i: " <<  i << " " << entities[i] + i <<std::endl;
        // auto start2 = std::chrono::high_resolution_clock::now();
        rmrt.insert(entities[i] + i, entities.begin(), i + 1);
        // entities.insert(entities.begin()+2*i + 1, entities[i] + i);
        // auto end2 = std::chrono::high_resolution_clock::now();

        // insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        // if ((i + 1) % (N / 10) == 0)
        // {
        //     std::cout << "i: " << i << " N: " << N << std::endl;
        //     std::cout << "insert time: " << ((i + 1) / (N / 10)) << "0%: " << std::endl;
        // }
    }
    // std::cout<< "insert time1: " << insert_time / N << std::endl;
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time2: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start3 = std::chrono::high_resolution_clock::now();
    long long correct_num = 0;
    for (size_t i = 0; i < N; i++)
    {
        uint64_t begin = 0;
        uint64_t end = 0;
        uint64_t pos = 0;
        pos = rmrt.search(entities[i], begin, end);
        end = end >= rmrt.N ? rmrt.N - 1 : end;
        if (entities[pos] == entities[i])
        {
            correct_num += 1;
            continue;
        }
        else
        {
            uint64_t mid = (begin + end) / 2;
            uint64_t orig_begin = begin;
            uint64_t orig_end = end;
            while (begin <= end)
            {
                if (entities[mid] < entities[i])
                {
                    begin = mid + 1;
                }
                else if (entities[mid] > entities[i])
                {
                    end = mid - 1;
                }
                else
                {
                    break;
                }
                mid = (begin + end) / 2;
            }
            if (entities[mid] != entities[i])
            {
                std::cout << "pos: " << pos << std::endl;
                std::cout << "begin: " << orig_begin << std::endl;
                std::cout << "end: " << orig_end << std::endl;
                std::cout << "not found"
                          << " i: " << i << std::endl;
                break;
            }
        }
    }
    std::cout << "correct_num: " << correct_num << std::endl;
    auto end3 = std::chrono::high_resolution_clock::now();
    std::cout << "final time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count() / entities.size() << std::endl;
    std::cout << std::endl;
    // N = 10000;
    // uint64_t insert_time = 0;

    std::cout << std::endl;
}

void exe_rmrt_lr(std::vector<uint64_t> entities, uint64_t page_size, uint64_t threshold, string tag)
{
    uint64_t N = entities.size();
    RMRT_LR rmrt(page_size, threshold, tag, 0);
    rmrt.is_root = true;
    rmrt.N = N;
    auto start = std::chrono::high_resolution_clock::now();
    // rmrt.build_recursively(entities.begin(), entities.size(), "" , 0);
    rmrt.build_recursively(entities);
    // N = 10000;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "build time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;
    std::cout << "size: " << rmrt.get_size() << std::endl;
    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        uint64_t start = 0;
        uint64_t end = 0;
        rmrt.search(entities[i], start, end);
    }
    uint64_t insert_time = 0;
    for (size_t i = 1; i <= N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        rmrt.insert(entities[i - 1] + i, i);
        auto end2 = std::chrono::high_resolution_clock::now();
        uint64_t temp = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        // if (temp < 1000)
        // {
        // }
        insert_time += temp;

        // std::cout << "insert time--------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() << " i: " << i << std::endl;
        if (i % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "0%: " << insert_time / i << std::endl;
        }
    }
    std::cout << "insert time1: " << insert_time / N << std::endl;
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time2: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start3 = std::chrono::high_resolution_clock::now();
    long long correct_num = 0;
    for (size_t i = 0; i < N; i++)
    {
        uint64_t begin = 0;
        uint64_t end = 0;
        uint64_t pos = 0;
        if (!rmrt.search_after_insertion(entities[i]))
        {
            std::cout << "not found"
                      << " i: " << i << std::endl;
            std::cout << "entities[i]" << entities[i] << std::endl;
            break;
        }
    }
    // std::cout<< "correct_num: " << correct_num << std::endl;
    auto end3 = std::chrono::high_resolution_clock::now();
    std::cout << "final time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count() / entities.size() << std::endl;
    std::cout << std::endl;
}

void exe_PGM1(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 64 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<64> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    // auto start = std::chrono::high_resolution_clock::now();
    // for (size_t i = 0; i < N; i++)
    // {
    //     pgm1.find(entities[i]);
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "0%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_PGM2(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 128 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<128> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    // auto start = std::chrono::high_resolution_clock::now();
    // for (size_t i = 0; i < N; i++)
    // {
    //     pgm1.find(entities[i]);
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_PGM3(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 256 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<256> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    // auto start = std::chrono::high_resolution_clock::now();
    // for (size_t i = 0; i < N; i++)
    // {
    //     pgm1.find(entities[i]);
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_PGM4(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 512 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<512> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_PGM5(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 1024 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<1024> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.insert(entities[i] + i, N + i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_PGM6(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 2048 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<2048> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.insert(entities[i] + i, N + i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_PGM7(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 4096 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<4096> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_PGM8(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 8192 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<8192> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_PGM9(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 16384 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<16384> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_PGM0(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "PGM branch: " << 16 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    PGMX<16> pgm1;
    pgm1.build(entities);
    uint64_t N = entities.size();
    // auto start = std::chrono::high_resolution_clock::now();
    // std::cout << "begin lookup: " << std::endl;
    // for (size_t i = 0; i < N; i++)
    // {
    //     pgm1.find(entities[i]);
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        pgm1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "0%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        pgm1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

// void exe_PGM11(std::vector<uint64_t> entities)
// {
//     uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
//     PGMX<65536> pgm1;
//     pgm1.build(entities);
//     uint64_t N = entities.size();
//     auto start = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         pgm1.find(entities[i]);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

//     auto start1 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         pgm1.insert(entities[i] + i, N + i);
//     }
//     auto end1 = std::chrono::high_resolution_clock::now();
//     std::cout<< "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

//     auto start2 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         pgm1.find(entities[i]+i);
//     }
//     auto end2 = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
// }

// void exe_PGM12(std::vector<uint64_t> entities)
// {
//     uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
//     PGMX<131072> pgm1;
//     pgm1.build(entities);
//     uint64_t N = entities.size();
//     auto start = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         pgm1.find(entities[i]);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

//     auto start1 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         pgm1.insert(entities[i] + i, N + i);
//     }
//     auto end1 = std::chrono::high_resolution_clock::now();
//     std::cout<< "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

//     auto start2 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         pgm1.find(entities[i]+i);
//     }
//     auto end2 = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
// }

// void exe_PGM13(std::vector<uint64_t> entities)
// {
//     uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
//     PGMX<262144> pgm1;
//     pgm1.build(entities);
//     uint64_t N = entities.size();
//     auto start = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         pgm1.find(entities[i]);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

//     auto start1 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         pgm1.insert(entities[i] + i, N + i);
//     }
//     auto end1 = std::chrono::high_resolution_clock::now();
//     std::cout<< "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

//     auto start2 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         pgm1.find(entities[i]+i);
//     }
//     auto end2 = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
// }

void exe_Btree0(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 8 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<8> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree1(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 64 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<64> btree1;
    auto start_build = std::chrono::high_resolution_clock::now();
    btree1.build(entities);
    auto end_build = std::chrono::high_resolution_clock::now();
    std::cout << "build time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_build - start_build).count() << std::endl;
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree2(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 128 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<128> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree3(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 256 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<256> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree4(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 512 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<512> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree5(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 1024 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<1024> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree6(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 2048 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<2048> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree7(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 4096 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<4096> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree8(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 8192 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<8192> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree9(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 16384 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<16384> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

void exe_Btree10(std::vector<uint64_t> entities)
{
    uint64_t insert_time = 0;
    std::cout << "btree branch: " << 32768 << std::endl;
    uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    Btree<32768> btree1;
    btree1.build(entities);
    uint64_t N = entities.size();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        btree1.insert(entities[i] + i, N + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0)
        {
            std::cout << "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++)
    {
        btree1.find(entities[i] + i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
}

// void exe_Btree11(std::vector<uint64_t> entities)
// {
//     uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
//     Btree<65536> btree1;
//     btree1.build(entities);
//     uint64_t N = entities.size();
//     auto start = std::chrono::high_resolutikon_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         btree1.find(entities[i]);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

//     auto start1 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         btree1.insert(entities[i] + i, N + i);
//     }
//     auto end1 = std::chrono::high_resolution_clock::now();
//     std::cout<< "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

//     auto start2 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         btree1.find(entities[i]+i);
//     }
//     auto end2 = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
// }

// void exe_Btree12(std::vector<uint64_t> entities)
// {
//     uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
//     Btree<131072> btree1;
//     btree1.build(entities);
//     uint64_t N = entities.size();
//     auto start = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         btree1.find(entities[i]);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

//     auto start1 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         btree1.insert(entities[i] + i, N + i);
//     }
//     auto end1 = std::chrono::high_resolution_clock::now();
//     std::cout<< "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

//     auto start2 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         btree1.find(entities[i]+i);
//     }
//     auto end2 = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
// }

// void exe_Btree13(std::vector<uint64_t> entities)
// {
//     uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
//     Btree<262144> btree1;
//     btree1.build(entities);
//     uint64_t N = entities.size();
//     auto start = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         btree1.find(entities[i]);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / N << std::endl;

//     auto start1 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         btree1.insert(entities[i] + i, N + i);
//     }
//     auto end1 = std::chrono::high_resolution_clock::now();
//     std::cout<< "insert time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

//     auto start2 = std::chrono::high_resolution_clock::now();
//     for(size_t i = 0; i < N; i++)
//     {
//         btree1.find(entities[i]+i);
//     }
//     auto end2 = std::chrono::high_resolution_clock::now();
//     std::cout<< "lookup time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() / N << std::endl;
// }

int main(int argc, char **argv)
{

    Net::load_trained_models("/home/research/code/SOSD/pre-train/trained_models/nn/4/0.3/");
    Net::load_trained_lr_models("/home/research/code/SOSD/pre-train/trained_models/linear/0.3/");

    // Net::load_trained_models("/home/research/code/SOSD/pre-train/trained_models/nn/4/0.2/");
    // Net::load_trained_lr_models("/home/research/code/SOSD/pre-train/trained_models/linear/0.2/");

    // Net::load_trained_models("/home/research/code/SOSD/pre-train/trained_models/nn/4/0.3/");
    // Net::load_trained_lr_models("/home/research/code/SOSD/pre-train/trained_models/linear/0.3/");

    // Net::load_trained_models("/home/research/code/SOSD/pre-train/trained_models/nn/4/0.4/");
    // Net::load_trained_lr_models("/home/research/code/SOSD/pre-train/trained_models/linear/0.4/");

    // Net::load_trained_models("/home/research/code/SOSD/pre-train/trained_models/nn/4/0.5/");
    // Net::load_trained_lr_models("/home/research/code/SOSD/pre-train/trained_models/linear/0.5/");

    // SynData synData;
    // std::vector<uint64_t> data = synData.getData("../../data/skewed_200000000_3.0_scale_1.csv");
    // synData.writeData(data, "../../data/skewed_200m_3");
    // data = synData.getData("../../data/skewed_200000000_5.0_scale_1.csv");
    // synData.writeData(data, "../../data/skewed_200m_5");
    // data = synData.getData("../../data/skewed_200000000_7.0_scale_1.csv");
    // synData.writeData(data, "../../data/skewed_200m_7");
    // data = synData.getData("../../data/skewed_200000000_9.0_scale_1.csv");
    // synData.writeData(data, "../../data/skewed_200m_9");

    string dataset = argv[1];

    string file_name = "../../data/" + dataset;

    FileReader filereader(file_name);
    std::vector<uint64_t> data = filereader.getData();
    // std::cout << "data.size(): " << data.size() << std::endl;

    // cout<< "start_x: " << data[0] << endl;
    // cout<< "end_x: " << data[data.size() - 1] << endl;

    // exe_rmi(data, 1, dataset);
    // uint64_t branches[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    // exp_recorder.prediction_time: 38
    // exp_recorder.insert_time_inner: 302
    // exp_recorder.insert_time: 349
    // exp_recorder.rebuild_time: 112
    // insert time1: 555
    // insert time2: 574
    // exp_recorder.vector_visit_time: 47
    // exp_recorder.prediction_time: 38
    // exp_recorder.time: 2745
    // search time2: 2921

    uint64_t branches[] = {100000};
    for (size_t i = 0; i < 2; i++)
    {
        std::cout << "rmi branch: " << branches[i] << std::endl;
        exe_rmi(data, branches[i], dataset);
        break;
    }

    // std::cout << "rmrt_lr branch: " << 1000000 << " N: " << 1000000 << std::endl;
    // exe_rmrt_lr(data, 1000000, 1000000, dataset);
    // std::cout << std::endl;
    // std::cout << "rmrt_lr branch: " << 1000000 << " N: " << 10000 << std::endl;
    // exe_rmrt_lr(data, 1000000, 10000, dataset);
    // std::cout << std::endl;
    // std::cout << "rmrt_lr branch: " << 1000000 << " N: " << 10000 << std::endl;
    // exe_rmrt_lr(data, 1000000, 6000000, dataset);
    // std::cout << "rmrt_lr branch: " << 1000000 << " N: " << 10000 << std::endl;
    // exe_rmrt_lr(data, 1000000, 1000, dataset);
    // std::cout << "rmrt_lr branch: " << 1000000 << " N: " << 10000 << std::endl;
    // exe_rmrt_lr(data, 1000000, 100, dataset);

    // exe_PGM0(data);
    // exe_PGM1(data);
    // exe_PGM2(data);
    // exe_PGM3(data);
    // exe_PGM4(data);
    // exe_PGM5(data);
    // exe_PGM6(data);
    // exe_PGM7(data);
    // exe_PGM8(data);
    // exe_PGM9(data);

    // exe_Btree0(data);
    // exe_Btree1(data);
    // exe_Btree2(data);
    // exe_Btree3(data);
    // exe_Btree4(data);
    // exe_Btree5(data);
    // exe_Btree6(data);
    // exe_Btree7(data);
    // exe_Btree8(data);
    // exe_Btree9(data);
    // exe_Btree10(data);

    return 0;
}