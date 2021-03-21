#include <iostream>
#include "include/RMI_NN.h"
#include "include/RMRT.h"
#include "include/RMRT_LR.h"
#include "utils/SynData.h"
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
    // RMRT rmrt(100, 1000000, tag);
    // rmrt.build_recursively(entities);
    RMI_NN rmi(true, tag, branch);
    // cout<< rmi.get_size() << endl;

    auto start = std::chrono::high_resolution_clock::now();
    if (branch == 1)
    {
        std::cout<< "build_one_layer" << std::endl;
        rmi.build_one_layer(entities);
    }
    else
    {
        std::cout<< "build_two_layer" << std::endl;
        rmi.build_two_layer(entities);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<< "build time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;
    std::cout<< "model size: " << rmi.get_size() << std::endl;
    uint64_t N = entities.size();
    // // N=20000;
    // auto start1 = std::chrono::high_resolution_clock::now();
    // uint64_t insert_time = 0;
    // for(size_t i = 0; i < N; i++)
    // {
    //     auto start2 = std::chrono::high_resolution_clock::now();
    //     rmi.insert_two_layer(entities[i] + i, entities.begin(), i + 1);
    //     auto end2 = std::chrono::high_resolution_clock::now();
    //     insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
    //     if ((i + 1) % (N / 10) == 0 )
    //     {
    //         std::cout<< "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
    //     }
    // }
    // auto end1 = std::chrono::high_resolution_clock::now();
    // std::cout<< "insert time2: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    long long correct_num = 0;
    auto start3 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < N; i++)
    {
        // auto start1 = chrono::high_resolution_clock::now();
        uint64_t begin = 0;
        uint64_t end = 0;
        uint64_t pos = 0;
      
        if (branch == 1)
        {
            pos = rmi.search_one_layer(entities[i], begin, end);
        }
        else
        {
            pos = rmi.search(entities[i], begin, end);
        }
        if (entities[pos] == entities[i])
        {
            correct_num += 1;
            continue;
        }
        else
        {
            // begin = pos > begin ? pos - begin : 0;
            // end = pos + end >= N ? N : pos + end;
            uint64_t mid = (begin + end) / 2;
            
            // if (end - begin > 100000)
            // {
            //     std::cout<< "i: " << i << " pos: " << pos << std::endl;
            //     std::cout<< "gap: " << (end - begin) << " begin: " << begin << " end: " << end << std::endl;
            // }
            
            uint64_t orig_begin = begin;
            uint64_t orig_end = end;
            while(begin <= end) {
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
                std::cout<< "begin: " << orig_begin << std::endl;
                std::cout<< "end: " << orig_end << std::endl;
                std::cout<< "not found" << " i: " << i << std::endl;
                break;
            }
        }
    }
    std::cout<< "correct_num: " << correct_num << std::endl;
    auto end3 = std::chrono::high_resolution_clock::now();
    std::cout<< "final time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count() / entities.size() << std::endl;

}

void exe_rmrt(std::vector<uint64_t> entities, uint64_t page_size, uint64_t threshold, string tag)
{
    uint64_t N = entities.size();
    RMRT rmrt(page_size, threshold, tag, 0);
    rmrt.is_root = true;
    rmrt.N = N;
    auto start = std::chrono::high_resolution_clock::now();
    // rmrt.build_recursively(entities.begin(), entities.size(), "" , 0);
    rmrt.build_recursively(entities, "" , 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<< "build time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;
    std::cout<< "size: " << rmrt.get_size() << std::endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    uint64_t insert_time = 0;
    for(size_t i = 0; i < N; i++)
    {
        // std::cout << "i: " <<  i << " " << entities[i] + i <<std::endl;
        auto start2 = std::chrono::high_resolution_clock::now();
        rmrt.insert(entities[i] + i, entities.begin(), i + 1);
        // entities.insert(entities.begin()+2*i + 1, entities[i] + i);
        auto end2 = std::chrono::high_resolution_clock::now();
        insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        if ((i + 1) % (N / 10) == 0 )
        {
            std::cout<< "insert time: " << ((i + 1) / (N / 10)) << "%: " << insert_time / i << std::endl;
        }
    }
    // std::cout<< "insert time1: " << insert_time / N << std::endl;
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout<< "insert time2: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() / N << std::endl;

    auto start3 = std::chrono::high_resolution_clock::now();
    long long correct_num = 0;
    for(size_t i = 0; i < N; i++)
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
            while(begin <= end) 
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
                std::cout<< "pos: " << pos << std::endl;
                std::cout<< "begin: " << orig_begin << std::endl;
                std::cout<< "end: " << orig_end << std::endl;
                std::cout<< "not found" << " i: " << i << std::endl;
                break;
            }
        }
    }
    std::cout<< "correct_num: " << correct_num << std::endl;
    auto end3 = std::chrono::high_resolution_clock::now();
    std::cout<< "final time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count() / entities.size() << std::endl;
    std::cout << std::endl;
    // N = 10000;
    // uint64_t insert_time = 0;
    
    std::cout<< std::endl;
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
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<< "build time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;
    std::cout<< "size: " << rmrt.get_size() << std::endl;

    // auto start3 = std::chrono::high_resolution_clock::now();
    // long long correct_num = 0;
    // for(size_t i = 0; i < N; i++)
    // {
    //     uint64_t begin = 0;
    //     uint64_t end = 0;
    //     uint64_t pos = 0;
    //     pos = rmrt.search(entities[i], begin, end);
    //     end = end >= rmrt.N ? rmrt.N - 1 : end;
    //     if (entities[pos] == entities[i])
    //     {
    //         correct_num += 1;
    //         continue;
    //     }
    //     else
    //     {
    //         uint64_t mid = (begin + end) / 2;
    //         uint64_t orig_begin = begin;
    //         uint64_t orig_end = end;
    //         while(begin <= end) 
    //         {
    //             if (entities[mid] < entities[i]) 
    //             {
    //                 begin = mid + 1;
    //             } 
    //             else if (entities[mid] > entities[i])
    //             {
    //                 end = mid - 1;
    //             }
    //             else
    //             {
    //                 break;
    //             }
    //             mid = (begin + end) / 2;
    //         }
    //         // std::cout<< "pos: " << pos << std::endl;
    //         // std::cout<< "begin: " << orig_begin << std::endl;
    //         // std::cout<< "end: " << orig_end << std::endl;
    //         if (entities[mid] != entities[i])
    //         {
    //             std::cout<< "pos: " << pos << std::endl;
    //             std::cout<< "begin: " << orig_begin << std::endl;
    //             std::cout<< "end: " << orig_end << std::endl;
    //             std::cout<< "not found" << " i: " << i << std::endl;
    //             break;
    //         }
    //     }
    // }
    // // std::cout<< "correct_num: " << correct_num << std::endl;
    // auto end3 = std::chrono::high_resolution_clock::now();
    // std::cout<< "final time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count() / entities.size() << std::endl;
    // std::cout << std::endl;
}

int main(int argc, char **argv) {

    Net::load_trained_models("/home/research/code/SOSD/pre-train/trained_models/nn/4/0.1/");
    Net::load_trained_lr_models("/home/research/code/SOSD/pre-train/trained_models/linear/0.1/");

    string dataset = argv[1];

    string file_name  = "../../data/" + dataset;

    FileReader filereader(file_name);
    std::vector<uint64_t> data = filereader.getData();
    std::cout<< "data.size(): " << data.size() << std::endl;

    uint64_t branches[] = {1024, 8192}; 
    for (size_t i = 0; i < 2; i++)
    {
        std::cout<< "rmi branch: " << branches[i] << std::endl;
        exe_rmi(data, branches[i], dataset);
        break;
    }
    return 0;
}