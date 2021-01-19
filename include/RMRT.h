#ifndef RMRT_CPP
#define RMRT_CPP

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

#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <dirent.h>

class RMRT
{
private:

public:
    uint64_t page_size;
    uint64_t real_page_size;
    // string PATH = "./pre-train/trained_models/nn/4/";
    string PATH = "../../pre-train/trained_models/nn/4/";
    
    string tag;

    uint64_t N;

    uint64_t threshold;

    uint64_t cardinality;

    std::vector<RMRT> children;

    std::vector<uint64_t> records;

    uint64_t max_error = 0;
    uint64_t min_error = 0;

    uint64_t start_y;

    uint64_t start_x;    
    uint64_t key_gap;

    bool is_leaf = false;
    bool is_root = false;

    std::shared_ptr<Net> net;

    // uint64_t insert_number_bound;
    // uint64_t insert_number;
    
    // bool build_recursively(std::vector<uint64_t>, string, uint64_t);

    // RMRT(int, uint64_t, string, uint64_t);

    // uint64_t search(uint64_t, uint64_t&, uint64_t&);

    // uint64_t get_size();

    RMRT()
    {}

    RMRT(int page_size, uint64_t threshold, string tag, uint64_t start_y)
    {
        this->page_size = page_size;
        this->real_page_size = page_size;
        this->threshold = threshold;
        this->tag = tag;
        this->start_y = start_y;
    }

    uint64_t get_size() const
    {
        int layer1_weights_num = Constants::WIDTH;
        int layer1_bias_num = Constants::WIDTH;
        int layer2_weights_num = Constants::WIDTH;
        int layer2_bias_num = 1;
        if (is_leaf)
        {
            int error_num = 2;
            int edge_value = 2;
            return sizeof(double) * (layer1_weights_num + layer1_bias_num + layer2_weights_num + layer2_bias_num) + sizeof(double) * (error_num + edge_value); 
        }
        else
        {
            uint64_t result = sizeof(double) * (layer1_weights_num + layer1_bias_num + layer2_weights_num + layer2_bias_num);
            // std::cout << "children.size(): " << children.size() << std::endl;
            for(RMRT rmrt : children)
            {
                // std::cout << "rmrt.get_size(): " << rmrt.get_size() << std::endl;
                result += rmrt.get_size();
            }
            return result;
        }
    }

    uint64_t search(uint64_t key, uint64_t& start, uint64_t& end)
    {
        // double search_key = double;
        // std::cout<< "key: " << key << std::endl;
        // std::cout<< "start_x: " << start_x << std::endl;
        // std::cout<< "key_gap: " << key_gap << std::endl;
        // if (key >= start_x && (start_x + key_gap) >= start_x)
        // {
        //     std::cout<< "yes!" << std::endl;
        // }

        // net->printParameters_Double();
        double res = net->predict_Double((key - start_x) * 1.0 / key_gap);
        if (is_leaf)
        {
            long long pos = (long long) (res * cardinality);
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
            int pos = (int) (res * real_page_size);
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

    bool build_recursively(std::vector<uint64_t> data, string prefix, uint64_t index)
    {
        cardinality = data.size();
        // std::cout<<"build_recursively:" <<cardinality<< std::endl;
        start_x = data[0];
        key_gap = data[cardinality - 1] - start_x;

        net = std::make_shared<Net>(start_x, key_gap, start_y, cardinality);
        string model_name = prefix + to_string(index);

        string path = "./torch_models/" + tag + "/RMRT/" + to_string(page_size) + "_"+ to_string(threshold) + "/" + model_name + ".pt";
    
        std::ifstream fin(path);
        // force enable model reuse
        // if (true)
        // {
        Histogram histogram(data);
        string model_path;
        double distance;
        if(net->is_model_reusable(histogram, 1.0, model_path, distance))
        {
            // net = net->models[model_path];
            // net->start_x = start_x;
            // net->key_gap = key_gap;
            // net->start_y = start_y;
            // net->cardinality = cardinality;
            // net->reuse(model_path);
            if (prefix == "rebuild")
            {
                net = net->models[model_path];
                net->start_x = start_x;
                net->x_gap = key_gap;
                net->start_y = start_y;
                net->y_gap = cardinality;
                // auto start1 = std::chrono::high_resolution_clock::now();
                // torch::load(net, (PATH + model_path + ".pt"));
                // auto end1 = std::chrono::high_resolution_clock::now();
                // std::cout<< "rebuild time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count()<< std::endl;
            }
            else
            {
                net = net->models[model_path];
                net->start_x = start_x;
                net->x_gap = key_gap;
                net->start_y = start_y;
                net->y_gap = cardinality;
                // torch::load(net, (PATH + model_path + ".pt"));
            }
            
            net->getParameters_Double();
            // std::cout<< "distance: " << distance << std::endl;
        }
        else
        {
            net->trainModel(data);
            torch::save(net, path);
        }

        if(cardinality < threshold) 
        {
            records = data;
            is_leaf = true;
            net->insert_number = 0;
            double gap = Constants::THRESHOLD - distance;
            gap = gap > 0 ? gap : 0.1;
            net->insert_number_bound = gap * cardinality / (1 - gap);
            for (long long i = 0; i < cardinality; i++)
            {
                double res = net->predict_Double((data[i] - start_x) * 1.0 / key_gap);
                long long pos = (long long) (res * cardinality);
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
            real_page_size = cardinality / threshold * 4;
            if (is_root || real_page_size > page_size)
            {
                real_page_size = page_size;
            }

            std::vector<std::vector<uint64_t>> children_entities;
            for (size_t i = 0; i < real_page_size; i++)
            {
                std::vector<uint64_t> children_entities_item;
                children_entities.push_back(children_entities_item);
            }
            for (uint64_t i = 0; i < cardinality; i++)
            {
                double res = net->predict_Double((data[i] - start_x) * 1.0 / key_gap);
                uint64_t pos = (long long) (res * real_page_size);
                if (pos < 0)
                {
                    pos = 0;
                }
                if (pos >= real_page_size)
                {
                    pos = real_page_size - 1;
                }
                children_entities[pos].push_back(data[i]);
            }

            std::vector<uint64_t> start_ys;
            uint64_t size_sum = 0;
            for (int i = 0; i < real_page_size; i++)
            {
                start_ys.push_back(size_sum);
                size_sum += children_entities[i].size();
            }

            for (size_t k = 0; k < children_entities.size(); k++)
            {
                // std::cout<< "children_entities[k].size(): " << children_entities[k].size() << std::endl;
                RMRT rmrt(page_size, threshold, tag, start_y + start_ys[k]);
                if (children_entities[k].size() == cardinality)
                {
                    std::cout<< "return false" << std::endl;
                    string path = "./torch_models/" + tag + "/RMRT/" + to_string(page_size) + "_"+ to_string(threshold) + "/" + model_name + ".pt";
                    char path_char[path.size() + 1];
                    strcpy(path_char, path.c_str());
                    remove(path_char);
                    return false;
                }
                if (children_entities[k].size() == 0)
                {
                    children.push_back(rmrt);
                    // children.push_back(NULL);
                    continue;
                }
                // bool is_force = false;
                // while (!rmrt.build_recursively(children_entities[k], model_name + "_", k, is_force))
                // {
                //     is_force = true;
                // }
                
                if(rmrt.build_recursively(children_entities[k], model_name + "_", k))
                {
                    children.push_back(rmrt);
                }
                else
                {
                    k--;
                }
            }
            return true;
        }
    }

    // bool build_recursively(std::vector<uint64_t>::iterator begin, uint64_t data_size, string prefix, uint64_t index)
    // {
    //     cardinality = data_size;
    //     std::cout<<"build_recursively:" <<cardinality<< std::endl;
    //     start_x = *begin;
    //     key_gap = *(begin + cardinality - 1) - start_x;

    //     net = std::make_shared<Net>(start_x, key_gap, start_y, cardinality);
    //     string model_name = prefix + to_string(index);

    //     string path = "./torch_models/" + tag + "/RMRT/" + to_string(page_size) + "_"+ to_string(threshold) + "/" + model_name + ".pt";
    
    //     std::ifstream fin(path);
    //     // force enable model reuse
    //     // if (true)
    //     // {
    //     // Histogram histogram(data);
    //     Histogram histogram(begin, data_size);
    //     std::cout<<"dfdfd"<<std::endl;
    //     string model_path;
    //     double distance;
    //     if(net->is_model_reusable(histogram, 1.0, model_path, distance))
    //     {
    //         // net = net->models[model_path];
    //         // net->start_x = start_x;
    //         // net->key_gap = key_gap;
    //         // net->start_y = start_y;
    //         // net->cardinality = cardinality;
    //         // net->reuse(model_path);
    //         torch::load(net, (PATH + model_path + ".pt"));
    //         net->getParameters_Double();
    //         // std::cout<< "distance: " << distance << std::endl;
    //     }
    //     // else
    //     // {
    //     //     net->trainModel(data);
    //     //     torch::save(net, path);
    //     // }

    //     if(cardinality < threshold) 
    //     {
    //         is_leaf = true;
    //         insert_number_bound = (distance - Constants::THRESHOLD) * cardinality / (1 + Constants::THRESHOLD - distance);
    //         // std::cout<< "insert_number_bound: " << insert_number_bound << std::endl;

    //         for (long long i = 0; i < cardinality; i++)
    //         {
    //             double res = net->predict_Double((*(begin + i) - start_x) * 1.0 / key_gap);
    //             long long pos = (long long) (res * cardinality);
    //             if (pos < 0)
    //             {
    //                 pos = 0;
    //             }
    //             if (pos >= cardinality)
    //             {
    //                 pos = cardinality - 1;
    //             }
    //             long long error = i - pos;
    //             if (error > 0)
    //             {
    //                 if (error > max_error)
    //                 {
    //                     max_error = error;
    //                 }
    //             }
    //             else
    //             {
    //                 if (-error > min_error)
    //                 {
    //                     min_error = -error;
    //                 }
    //             }
    //         }
    //         return true;
    //     }
    //     else
    //     {
    //         real_page_size = cardinality / threshold * 4;
    //         if (real_page_size > page_size)
    //         {
    //             real_page_size = page_size;
    //         }

    //         std::vector<std::vector<uint64_t>> children_entities;
    //         for (size_t i = 0; i < real_page_size; i++)
    //         {
    //             std::vector<uint64_t> children_entities_item;
    //             children_entities.push_back(children_entities_item);
    //         }
    //         for (uint64_t i = 0; i < cardinality; i++)
    //         {
    //             double res = net->predict_Double((*(begin + i) - start_x) * 1.0 / key_gap);
    //             uint64_t pos = (long long) (res * real_page_size);
    //             if (pos < 0)
    //             {
    //                 pos = 0;
    //             }
    //             if (pos >= real_page_size)
    //             {
    //                 pos = real_page_size - 1;
    //             }
    //             children_entities[pos].push_back(*(begin + i));
    //         }

    //         std::vector<uint64_t> start_ys;
    //         uint64_t size_sum = 0;
    //         for (int i = 0; i < real_page_size; i++)
    //         {
    //             start_ys.push_back(size_sum);
    //             size_sum += children_entities[i].size();
    //         }

    //         for (size_t k = 0; k < children_entities.size(); k++)
    //         {
    //             // std::cout<< "children_entities[k].size(): " << children_entities[k].size() << std::endl;
    //             RMRT rmrt(page_size, threshold, tag, start_y + start_ys[k]);
    //             if (children_entities[k].size() == cardinality)
    //             {
    //                 std::cout<< "return false" << std::endl;
    //                 string path = "./torch_models/" + tag + "/RMRT/" + to_string(page_size) + "_"+ to_string(threshold) + "/" + model_name + ".pt";
    //                 char path_char[path.size() + 1];
    //                 strcpy(path_char, path.c_str());
    //                 remove(path_char);
    //                 return false;
    //             }
    //             if (children_entities[k].size() == 0)
    //             {
    //                 children.push_back(rmrt);
    //                 // children.push_back(NULL);
    //                 continue;
    //             }
    //             // bool is_force = false;
    //             // while (!rmrt.build_recursively(children_entities[k], model_name + "_", k, is_force))
    //             // {
    //             //     is_force = true;
    //             // }
                
    //             if(rmrt.build_recursively(children_entities[k].begin(), children_entities[k].size(), model_name + "_", k))
    //             {
    //                 children.push_back(rmrt);
    //             }
    //             else
    //             {
    //                 k--;
    //             }
    //         }
    //         return true;
    //     }
    // }

    void update_rebuild(std::vector<uint64_t> data)
    {
        cardinality = data.size();
        start_x = data[0];
        key_gap = data[cardinality - 1] - start_x;
        net = std::make_shared<Net>(start_x, key_gap, start_y, cardinality);
        Histogram histogram(data);
        string model_path;
        double distance;
        net->is_model_reusable(histogram, 1.0, model_path, distance);
        // auto start2 = std::chrono::high_resolution_clock::now();
        net = net->models[model_path];
        net->start_x = start_x;
        net->x_gap = key_gap;
        net->start_y = start_y;
        net->y_gap = cardinality;
        // auto end2 = std::chrono::high_resolution_clock::now();
        // std::cout<< "rebuild time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count()<< std::endl;
        net->getParameters_Double();
        // net->printParameters_Double();
        net->insert_number = 0;
        double gap = Constants::THRESHOLD - distance;
        gap = gap > 0 ? gap : 0.1;
        net->insert_number_bound = gap * cardinality / (1 - gap);
        // std::cout<< "insert_number_bound: " << insert_number_bound << std::endl;
        net->min_error = cardinality * distance + (-net->model_infos[model_path].min_err * 100000) * 100000 / cardinality;
        net->max_error = cardinality * distance + (net->model_infos[model_path].max_err * 100000) * 100000 / cardinality;
        // std::cout<< "min_err: " << net->min_error << std::endl;
        // std::cout<< "max_err: " << net->max_error << std::endl;
        // std::cout<< "insert_number_bound: " << insert_number_bound << std::endl;
    }

    void insert(uint64_t insert_data, std::vector<uint64_t>::iterator begin, uint64_t inserted_num)
    {

        // auto start1 = std::chrono::high_resolution_clock::now();
        double res = net->predict_Double((insert_data - start_x) * 1.0 / key_gap);
        // auto end1 = std::chrono::high_resolution_clock::now();
        // std::cout<< "time1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() << std::endl;
        if (is_leaf)
        {
            auto start2 = std::chrono::high_resolution_clock::now();
            //TODO calibration
            uint64_t start = start_y;
            uint64_t end = start_y + cardinality + inserted_num;
            uint64_t mid = (start + end) / 2;
            if (*(begin + start_y) != start_x && start <= end)
            {
                uint64_t temp = *(begin + mid);
                while(temp != start_x) 
                {
                    if (temp < start_x) 
                    {
                        start = mid + 1;
                    } 
                    else if (temp > start_x)
                    {
                        end = mid - 1;
                    }
                    else
                    {
                        break;
                    }
                    mid = (start + end) / 2;
                    temp = *(begin + mid);
                }
            }
            auto end2 = std::chrono::high_resolution_clock::now();
            // std::cout<< "calibration time2: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() << std::endl;

            // auto start3 = std::chrono::high_resolution_clock::now();
            //TODO insert
            net->insert_number++;
            uint64_t pos = (uint64_t) (res * cardinality);
            start = start_y;
            end = start_y + cardinality;
            mid = (start + end) / 2;
            // std::cout<< "start: " << *(begin + start) << " end: " << *(begin + end) << " insert_data: " << insert_data << std::endl;
            while(start < end) {
                // std::cout<< "start: " << start << " end: " << end << " mid: " << mid << std::endl;
                if (*(begin + mid) <= insert_data) 
                {
                    if (*(begin + mid + 1) >= insert_data) 
                    {
                        break;
                    }
                    start = mid + 1;
                } 
                else if (*(begin + mid) > insert_data)
                {
                    if (*(begin + mid - 1) <= insert_data)
                    {
                        mid -= 1;
                        break;
                    }
                    end = mid - 1;
                }
                mid = (start + end) / 2;
            }
            // start_y = mid;
            // auto end3 = std::chrono::high_resolution_clock::now();
            // std::cout<< "insert time2: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count() << std::endl;

            if (net->insert_number >= net->insert_number_bound)
            {
                // TODO rebuild!!!
                // std::cout<<"rebuild!!!" << std::endl;
                // auto start4 = std::chrono::high_resolution_clock::now();
                update_rebuild(records);
                // build_recursively(records, "", 0);
                // auto end4 = std::chrono::high_resolution_clock::now();
                // std::cout<< "rebuild : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - start4).count()<< std::endl;
            }
        }
        else
        {
            int pos = (int) (res * real_page_size);
            if (pos < 0)
            {
                pos = 0;
            }
            if (pos >= real_page_size)
            {
                pos = real_page_size - 1;
            }
            children[pos].insert(insert_data, begin, inserted_num);
        }
    }
    
};


#endif