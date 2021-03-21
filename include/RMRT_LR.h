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

    int level = 0;
    int index = 0;

    double a;
    double b;

    int insert_number_bound = 0;
    int insert_number = 0;

    // bool is_print = false;

    RMRT_LR()
    {
    }

    RMRT_LR(int page_size, uint64_t threshold, string tag, uint64_t start_y)
    {
        this->page_size = page_size;
        this->real_page_size = page_size;
        this->threshold = threshold;
        this->tag = tag;
        this->start_y = start_y;
        this->level = 0;
        this->index = 0;
    }

    RMRT_LR(int page_size, uint64_t threshold, string tag, uint64_t start_y, int level, int index)
    {
        this->page_size = page_size;
        this->real_page_size = page_size;
        this->threshold = threshold;
        this->tag = tag;
        this->start_y = start_y;
        this->level = level;
        this->index = index;
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
            for (RMRT_LR rmrt : children)
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

    uint64_t search(uint64_t key, uint64_t &start, uint64_t &end)
    {
        double res = predict(key);
        if (is_leaf)
        {
            long long pos = (long long)(res);
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
            int pos = (int)(res * real_page_size / cardinality);
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
        // if(data.size() > 100)
        // {
            // std::cout<<"build_recursively:" << cardinality << " level: " << level << " index: " << index << std::endl;
        // }
        start_x = data[0];
        key_gap = data[cardinality - 1] - start_x;

        Histogram histogram(data);
        double distance;
        if (is_root)
        {
            // std::cout<< PATH << std::endl;
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
        if (cardinality < threshold)
        {
            records = data;
            is_leaf = true;
            double gap = Constants::THRESHOLD - distance;
            gap = gap > 0 ? gap : 0.1;
            for (long long i = 0; i < cardinality; i++)
            {
                // double res = predict((data[i] - start_x) * 1.0 / key_gap);
                double res = predict(data[i]);
                long long pos = (long long)(res);
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
            is_leaf = false;
            // real_page_size = cardinality / threshold * 16;
            // if (is_root || real_page_size > page_size)
            // {
            //     real_page_size = page_size;
            // }
            // TODO can remove
            real_page_size = page_size;
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
                uint64_t pos = (long long)(res * real_page_size / cardinality);
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
                if (each_block_end_index[i] == 0)
                {
                    each_block_end_index[i] = each_block_end_index[i - 1];
                }
                // std::cout<< "y: " << (int)predict(data[each_block_end_index[i] - 1])* real_page_size / cardinality << std::endl;
                // start_ys.push_back(each_block_end_index[i]);
            }
            // std::cout<< "start_ys: " << start_ys << std::endl;
            // if (is_print)
            // {
            //     std::cout<< "each_block_end_index.size(): " << each_block_end_index.size() << std::endl;
            //     std::cout<< "each_block_end_index: " << each_block_end_index << std::endl;
            // }
            for (size_t k = 0; k < each_block_end_index.size() - 1; k++)
            {

                RMRT_LR rmrt(page_size, threshold, tag, start_y + each_block_end_index[k], level + 1, k);
                std::vector<uint64_t> children_entities_item(data.begin() + each_block_end_index[k], data.begin() + each_block_end_index[k + 1]);
                if (children_entities_item.size() == 0)
                {
                    rmrt.is_leaf = true;
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

    void update_rebuild(uint64_t inserted_num, std::vector<uint64_t> data)
    {
        // std::cout<<"------------------------------------update_rebuild: level: " << level << " index: " << index << std::endl;
        cardinality = data.size();
        std::sort(data.begin(), data.end());
        // std::cout<<"build_recursively:" <<cardinality<< std::endl;
        // start_x = data[0];
        // key_gap = data[cardinality - 1] - start_x;
        double p_a;
        double p_b;
        double x_0 = data[0];
        double x_1 = data[cardinality - 1];
        double y_0 = 0;
        double y_1 = cardinality - 1;
        a = (y_1 - y_0) / (x_1 - x_0);
        b = (y_0 - a * x_0 + y_1 - a * x_1) / 2;
        // std::cout<< "a: " << a << " b: " << b << std::endl;
        // std::cout<< "cardinality: " << cardinality << std::endl;
        is_leaf = false;
        // real_page_size = cardinality / threshold * 16;
        // if (is_root || real_page_size > page_size)
        // {
        //     real_page_size = page_size;
        // }
        // TODO can remove
        real_page_size = page_size;
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
            uint64_t pos = (long long)(res * real_page_size / cardinality);
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
            if (each_block_end_index[i] == 0)
            {
                each_block_end_index[i] = each_block_end_index[i - 1];
            }
            // std::cout<< "y: " << (int)predict(data[each_block_end_index[i] - 1])* real_page_size / cardinality << std::endl;
            // start_ys.push_back(each_block_end_index[i]);
        }
        // std::cout<< "start_ys: " << start_ys << std::endl;
        // if (is_print)
        // {
        //     std::cout<< "each_block_end_index.size(): " << each_block_end_index.size() << std::endl;
        //     std::cout<< "each_block_end_index: " << each_block_end_index << std::endl;
        // }
        for (size_t k = 0; k < each_block_end_index.size() - 1; k++)
        {
            RMRT_LR rmrt(page_size, threshold, tag, each_block_end_index[k], level + 1, k);
            std::vector<uint64_t> children_entities_item(data.begin() + each_block_end_index[k], data.begin() + each_block_end_index[k + 1]);
            // if (children_entities_item.size() <= 1)
            // {
            //     children.push_back(rmrt);
            //     continue;
            // }
            // rmrt.build_recursively(children_entities_item);
            rmrt.records = children_entities_item;
            rmrt.cardinality = children_entities_item.size();
            rmrt.is_leaf = true;
            children.push_back(rmrt);
        }
    }

    void insert(uint64_t insert_data, uint64_t inserted_num)
    {
        if (is_leaf)
        {
            // std::cout << "is_leaf: " << std::endl;
            // auto start2 = std::chrono::high_resolution_clock::now();
            // std::cout << "records size: " << records.size() << std::endl;
            is_leaf = true;
            if (records.size() == 0)
            {
                records.push_back(insert_data);
                return;
            }

            if (records[0] > insert_data)
            {
                // std::cout << "add first: " << std::endl;
                records.insert(records.begin(), insert_data);
            }
            else if (records[records.size() - 1] < insert_data)
            {
                // std::cout << "add last: " << std::endl;
                // auto start2 = std::chrono::high_resolution_clock::now();
                records.push_back(insert_data);
                // auto end2 = std::chrono::high_resolution_clock::now();
                // std::cout << "insert time last: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() << std::endl;
            }
            else
            {
                uint64_t start = 0;
                uint64_t finish = records.size() - 1;
                uint64_t mid = 0;
                // auto start1 = std::chrono::high_resolution_clock::now();
                while (start <= finish)
                {
                    mid = (start + finish) / 2;
                    uint64_t temp_0 = records[mid];
                    uint64_t temp_1 = records[mid + 1];
                    // std::cout << "temp_0: " << temp_0 << std::endl;
                    // std::cout << "temp_1: " << temp_1 << std::endl;
                    // std::cout << "mid: " << mid << std::endl;
                    if (temp_0 > insert_data)
                    {
                        finish = mid - 1;
                    }
                    else if (temp_1 < insert_data)
                    {
                        start = mid + 1;
                    }
                    else
                    {
                        // auto start2 = std::chrono::high_resolution_clock::now();
                        // std::cout << "before records: " << records << std::endl;
                        records.insert(records.begin() + mid + 1, insert_data);
                        // std::cout << "after records: " << records << std::endl;
                        // auto end2 = std::chrono::high_resolution_clock::now();
                        // if (inserted_num == 9978 || inserted_num == 9977 || inserted_num == 9979)
                        // {
                        //     std::cout << "records size: " << records.size() << std::endl;
                        //     std::cout << "insert time2 : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() << std::endl;
                        // }
                        break;
                    }
                }
                // auto end1 = std::chrono::high_resolution_clock::now();
                // if (inserted_num == 9978 || inserted_num == 9977 || inserted_num == 9979)
                // {
                //     std::cout << "insert time1 : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count() << std::endl;
                // }
            }
            // auto end2 = std::chrono::high_resolution_clock::now();
            if (records.size() > threshold)
            {
                // std::cout<< "records size: " << records.size() << std::endl;
                // build_recursively(records);
                // auto start2 = std::chrono::high_resolution_clock::now();
                update_rebuild(inserted_num, records);
                // auto end2 = std::chrono::high_resolution_clock::now();
                // std::cout << "rebuild time : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() << std::endl;
                // records.clear();
                // records.shrink_to_fit();
            }
            // auto end2 = std::chrono::high_resolution_clock::now();
            //     std::cout << "leaf-insert time : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() << std::endl;
        }
        else
        {
            // std::cout << "is non-leaf: " << std::endl;
            // auto start2 = std::chrono::high_resolution_clock::now();
            double res = predict(insert_data);
            int pos = (int)(res * real_page_size / cardinality);
            if (pos < 0)
            {
                pos = 0;
            }
            if (pos >= real_page_size)
            {
                pos = real_page_size - 1;
            }
            // if (inserted_num == 9978 || inserted_num == 9977 || inserted_num == 9979)
            // {
            //     std::cout << "non-leaf--------------: " << std::endl;
            //     std::cout << "inserted_num: " << inserted_num << std::endl;
            //     std::cout << "insert_data: " << insert_data << std::endl;
            //     std::cout << "level: " << level << std::endl;
            //     std::cout << "index: " << index << std::endl;
            //     std::cout << "res: " << res << std::endl;
            //     std::cout << "pos: " << pos << std::endl;
            //     std::cout << "cardinality: " << cardinality << std::endl;
            //     std::cout << "records.size(): " << records.size() << std::endl;
            //     std::cout << "children.size(): " << children.size() << std::endl;
            // }
            RMRT_LR *temp = &children[pos];
            temp->insert(insert_data, inserted_num);
            // auto end2 = std::chrono::high_resolution_clock::now();
            // std::cout << "non-leaf time : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() << std::endl;
        }
    }

    bool search_after_insertion(uint64_t key)
    {
        if (is_leaf)
        {
            // std::cout << "key: " << key << std::endl;
            // std::cout << "records: " << records << std::endl;

            uint64_t start = 0;
            uint64_t finish = records.size() - 1;
            uint64_t mid = 0;
            while (start <= finish)
            {
                mid = (start + finish) / 2;
                uint64_t temp = records[mid];
                // std::cout << "mid: " << mid << std::endl;
                // std::cout << "temp: " << temp << std::endl;
                if (temp == key)
                {
                    return true;
                }
                else if (temp < key)
                {
                    start = mid + 1;
                }
                else
                {
                    finish = mid - 1;
                }
            }
            return false;
        }
        else
        {
            double res = predict(key);
            int pos = (int)(res * real_page_size / cardinality);
            if (pos < 0)
            {
                pos = 0;
            }
            if (pos >= real_page_size)
            {
                pos = real_page_size - 1;
            }
            return children[pos].search_after_insertion(key);
        }
    }
};

#endif