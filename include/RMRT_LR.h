#ifndef RMRT_LR_CPP
#define RMRT_LR_CPP

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>
#include <stdio.h>

#include "../entities/Histogram.h"
#include "../utils/ModelTools.h"
#include "../utils/Constants.h"

#include <dirent.h>

class RMRT_LR
{
private:

public:
    uint64_t page_size;
    uint64_t real_page_size;
    string PATH = "./pre-train/trained_models/linear/0.1/";
    // string PATH = "../../pre-train/trained_models/linear/" + to_string(Constants::THRESHOLD) + "/";
    
    string tag;

    uint64_t N;

    uint64_t threshold;

    uint64_t cardinality;

    std::vector<RMRT_LR> children;

    std::vector<uint64_t> records;

    uint64_t max_error = 0;
    uint64_t min_error = 0;

    uint64_t start_y;

    uint64_t start_x;    
    uint64_t key_gap;

    bool is_leaf = false;
    bool is_root = false;

    double a;
    double b;

    RMRT_LR()
    {}

    RMRT_LR(int page_size, uint64_t threshold, string tag, uint64_t start_y)
    {
        this->page_size = page_size;
        this->real_page_size = page_size;
        this->threshold = threshold;
        this->tag = tag;
        this->start_y = start_y;
    }

    uint64_t get_size() const
    {
        if (is_leaf)
        {
            int error_num = 2;
            int edge_value = 2;
            return sizeof(double) + sizeof(double);
        }
        else
        {
            uint64_t result = sizeof(double) + sizeof(double);
            for(RMRT_LR rmrt : children)
            {
                result += rmrt.get_size();
            }
            return result;
        }
    }

    double predict(double x)
    {
        return a * x + b;
    }

    uint64_t search(uint64_t key, uint64_t& start, uint64_t& end)
    {
        double res = predict(key);
        if (is_leaf)
        {
            long long pos = (long long) (res);
            // std::cout<< "cardinality: " << cardinality << " pos:" << pos << std::endl;
            // std::cout<< "start_y: " << start_y << " min_error:" << min_error << " max_error:" << max_error << std::endl;
            if (pos < 0)
            {
                pos = 0;
            }
            if (pos >= cardinality)
            {
                pos = cardinality - 1;
            }
            start = pos > min_error ? pos + start_y - min_error : start_y;
            end = pos + start_y + max_error;
            return pos + start_y;
        }
        else
        {
            int pos = (int) (res * real_page_size / cardinality);
            if (pos < 0)
            {
                pos = 0;
            }
            if (pos >= real_page_size)
            {
                pos = real_page_size - 1;
            }
            return children[pos].search(key, start, end);
        }
    }

    bool build_recursively(std::vector<uint64_t> data)
    {
        cardinality = data.size();
        // std::cout<<"build_recursively:" <<cardinality<< std::endl;
        start_x = data[0];
        key_gap = data[cardinality - 1] - start_x;

        Histogram histogram(data);
        double distance;
        string model_path;
        if (is_root)
        {
            std::cout<< PATH << std::endl;
            Net::load_trained_lr_models(PATH);
        }
        double p_a;
        double p_b;
        double x_0 = data[0];
        double x_1 = data[cardinality - 1];
        double y_0 = 0;
        double y_1 = cardinality - 1;
        // if(Net::is_model_lr_reusable(histogram, 1.0, p_a, p_b, distance))
        // {
        //     // std::cout<< "p_a: " << p_a << " p_b: " << p_b << std::endl;
        //     // std::cout<< "distance: " << distance << std::endl;
        //     a = p_a * (y_1 - y_0) / (x_1 - x_0);
        // }
        // else
        // {
        //     a = (y_1 - y_0) / (x_1 - x_0);
        // }
        a = (y_1 - y_0) / (x_1 - x_0);
        b = (y_0 - a * x_0 + y_1 - a * x_1) / 2;
        // std::cout<< "a: " << a << " b: " << b << std::endl;
        // std::cout<< "cardinality: " << cardinality << std::endl;
        if(cardinality < threshold)
        {
            records = data;
            is_leaf = true;
            double gap = Constants::THRESHOLD - distance;
            gap = gap > 0 ? gap : 0.1;
            for (long long i = 0; i < cardinality; i++)
            {
                // double res = predict((data[i] - start_x) * 1.0 / key_gap);
                double res = predict(data[i]);
                long long pos = (long long) (res);
                if (pos < 0)
                {
                    pos = 0;
                }
                if (pos >= cardinality)
                {
                    pos = cardinality - 1;
                }
                long long error = i - pos;
                if (error > 0)
                {
                    if (error > max_error)
                    {
                        max_error = error;
                    }
                }
                else
                {
                    if (-error > min_error)
                    {
                        min_error = -error;
                    }
                }
            }
            return true;
        }
        else
        {
            real_page_size = cardinality / threshold * 16;
            if (is_root || real_page_size > page_size)
            {
                real_page_size = page_size;
            }

            std::vector<std::vector<uint64_t>> children_entities;
            std::vector<uint64_t> each_block_end_index(real_page_size + 1);
            for (size_t i = 0; i < real_page_size; i++)
            {
                std::vector<uint64_t> children_entities_item;
                children_entities.push_back(children_entities_item);
            }
            for (uint64_t i = 0; i < cardinality; i++)
            {
                double res = predict(data[i]);
                // double res = predict((data[i] - start_x) * 1.0 / key_gap);
                uint64_t pos = (long long) (res * real_page_size / cardinality);
                if (pos < 0)
                {
                    pos = 0;
                }
                if (pos >= real_page_size)
                {
                    pos = real_page_size - 1;
                }
                each_block_end_index[pos + 1] = i + 1;
                // children_entities[pos].push_back(data[i]);
            }
            each_block_end_index[real_page_size] = cardinality;
            // std::cout<< "each_block_end_index: " << each_block_end_index << std::endl;
            // std::vector<uint64_t> start_ys;
            // uint64_t size_sum = 0;
            // start_ys.push_back(0);
            for (int i = 1; i < real_page_size + 1; i++)
            {
                // start_ys.push_back(size_sum);
                // size_sum += children_entities[i].size();
                if(each_block_end_index[i] == 0)
                {
                    each_block_end_index[i] = each_block_end_index[i - 1];
                }
                // std::cout<< "y: " << (int)predict(data[each_block_end_index[i] - 1])* real_page_size / cardinality << std::endl;
                // start_ys.push_back(each_block_end_index[i]);
            }
            // std::cout<< "start_ys: " << start_ys << std::endl;
            // std::cout<< "each_block_end_index: " << each_block_end_index << std::endl;
            for (size_t k = 0; k < each_block_end_index.size() - 1; k++)
            {
                RMRT_LR rmrt(page_size, threshold, tag, start_y + each_block_end_index[k]);
                std::vector<uint64_t> children_entities_item(data.begin() + each_block_end_index[k], data.begin() + each_block_end_index[k + 1]);
                if (children_entities_item.size() == 0)
                {
                    children.push_back(rmrt);
                    continue;
                }
                rmrt.build_recursively(children_entities_item);
                children.push_back(rmrt);
            }
            // for (size_t k = 0; k < children_entities.size(); k++)
            // {
            //     RMRT_LR rmrt(page_size, threshold, tag, start_y + start_ys[k]);
            //     if (children_entities[k].size() == 0)
            //     {
            //         children.push_back(rmrt);
            //         continue;
            //     }
                
            //     if(rmrt.build_recursively(children_entities[k]))
            //     {
            //         children.push_back(rmrt);
            //     }
            //     else
            //     {
            //         k--;
            //     }
            // }
            return true;
        }
    }

};


#endif