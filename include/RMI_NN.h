#ifndef RMI_NN_CPP
#define RMI_NN_CPP

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

#include "../entities/Histogram.h"
#include "../utils/ModelTools.h"
#include "../utils/Constants.h"
#include "../utils/ExpRecorder.h"

#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

// struct ApproxPos {
//     size_t pos; ///< The approximate position of the key.
//     size_t lo;  ///< The lower bound of the range where the key can be found.
//     size_t hi;  ///< The upper bound of the range where the key can be found.
// };

// -ffast-math -Wall -Wfatal-errors -march=native -fopenmp
//  -ffast-math -Wall -Wfatal-errors -march=native

class RMI_NN
{
private:
    uint64_t N;
    std::set<string> reused_models;
    // string PATH = "./pre-train/trained_models/nn/4/" + to_string(Constants::THRESHOLD) + "/";
    string PATH = "/home/research/code/SOSD/pre-train/trained_models/nn/4/0.1/";
    // string PATH = "../../pre-train/trained_models/nn/4/" + to_string(Constants::THRESHOLD) + "/";
public:
    long time_cost_1 = 0;
    long time_cost_2 = 0;
    long time_cost_31 = 0;
    long time_cost_32 = 0;
    long time_cost_4 = 0;
    long time_cost_5 = 0;
    string tag;

    // RMI_NN(string, int);

    // RMI_NN(int);

    // RMI_NN();

    bool is_model_reuse = false;
    int width;

    std::vector<uint64_t> records;

    std::vector<std::vector<std::shared_ptr<Net>>> index;

    std::vector<std::shared_ptr<Net>> layer1_index;
    std::vector<std::shared_ptr<Net>> layer2_index;

    std::vector<std::vector<uint64_t>> stage2;

    // void build_two_layer(std::vector<uint64_t>);
    // void build_one_layer(std::vector<uint64_t>);
    // long long search(uint64_t, uint64_t&, uint64_t&);
    // long long search_one_layer(uint64_t, uint64_t&, uint64_t&);
    // size_t get_size() const;

    int branch;

    RMI_NN(bool is_model_reuse, string tag, int branch)
    {
        this->is_model_reuse = is_model_reuse;
        this->tag = tag;
        this->width = Constants::WIDTH;
        ;
        this->branch = branch;
    }

    RMI_NN(string tag, int branch)
    {
        this->tag = tag;
        this->width = Constants::WIDTH;
        ;
        this->branch = branch;
    }

    RMI_NN(bool is_model_reuse, int branch)
    {
        this->is_model_reuse = is_model_reuse;
        this->width = Constants::WIDTH;
        this->branch = branch;
    }

    RMI_NN(int branch)
    {
        this->width = Constants::WIDTH;
        this->branch = branch;
    }

    RMI_NN(bool is_model_reuse)
    {
        this->is_model_reuse = is_model_reuse;
        this->width = Constants::WIDTH;
    }

    RMI_NN()
    {
        this->width = Constants::WIDTH;
    }

    size_t get_size() const
    {
        // layer1 weights + bias  layer2 weights + bias
        int layer1_weights_num = width;
        int layer1_bias_num = width;
        int layer2_weights_num = width;
        int layer2_bias_num = 1;

        int layer1_model_num = 1;
        int layer2_model_num = branch;

        int error_num = 6;

        long long model_size = layer1_weights_num + layer1_bias_num + layer2_weights_num + layer2_bias_num;

        if (is_model_reuse)
        {
            // std::cout << "size of reused models: " << reused_models.size() << std::endl;
            return sizeof(double) * reused_models.size() + sizeof(double) * reused_models.size() * model_size + sizeof(double) * error_num * layer2_model_num;
        }
        else
        {
            return sizeof(double) * model_size * (layer1_model_num + layer2_model_num) + sizeof(double) * error_num * layer2_model_num;
        }
    }

    long long search_one_layer(uint64_t search_key, uint64_t &begin, uint64_t &end)
    {
        std::shared_ptr<Net> top = index[0][0];
        double search_key_double = double(search_key - top->start_x) / top->x_gap;
        long long pos = top->predict_Double(search_key_double) * top->y_gap;
        if (pos < 0)
        {
            pos = 0;
        }
        if (pos >= top->y_gap)
        {
            pos = top->y_gap - 1;
        }
        begin = pos > top->min_error ? pos - top->min_error : 0;
        end = pos + top->max_error >= top->y_gap ? top->y_gap - 1 : pos + top->max_error;
        return pos;
    }

    long long search(ExpRecorder &exp_recorder, uint64_t search_key, uint64_t &begin, uint64_t &end)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        std::shared_ptr<Net> top = index[0][0];
        double search_key_double = double(search_key - top->start_x) / top->x_gap;
        // std::cout << "search_key: " << search_key << std::endl;
        // std::cout << "top->start_x: " << top->start_x << std::endl;
        // std::cout << "top->x_gap: " << top->x_gap << std::endl;
        // std::cout << "search_key_double: " << search_key_double << std::endl;
        auto finish1 = std::chrono::high_resolution_clock::now();
        exp_recorder.vector_visit_time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish1 - start1).count();
        // std::cout<< "search : normalize 1------1------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish1 - start1).count() << std::endl;
        // time_cost_1 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish1 - start1).count();
        auto start2 = std::chrono::high_resolution_clock::now();
        long long pos = top->predict_Double(search_key_double) * branch;
        // std::cout << "pos: " << pos << std::endl;
        auto finish2 = std::chrono::high_resolution_clock::now();
        exp_recorder.prediction_time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish2 - start2).count();
        // std::cout<< "search prediction 1------2------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish2 - start2).count() << std::endl;
        // time_cost_2 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish2 - start2).count();
        if (pos < 0)
        {
            pos = 0;
        }
        else if (pos >= branch)
        {
            pos = branch - 1;
        }
        auto start31 = std::chrono::high_resolution_clock::now();
        // std::shared_ptr<Net> net = index[1][pos];
        std::shared_ptr<Net> net = layer2_index[pos];
        auto finish31 = std::chrono::high_resolution_clock::now();
        // std::cout<< "get model in layer2------31------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish31 - start31).count() << std::endl;
        exp_recorder.vector_visit_time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish31 - start31).count();
        // std::cout<< "pos: " << pos << std::endl;
        // std::cout<< "net->start_y: " << net->start_y << std::endl;
        // std::cout<< "search_key: " << search_key << std::endl;
        // auto start32 = std::chrono::high_resolution_clock::now();
        search_key_double = double(search_key - net->start_x) / net->x_gap;
        // auto finish32 = std::chrono::high_resolution_clock::now();
        // std::cout<< "search : normalize 2------32------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish32 - start32).count() << std::endl;
        // time_cost_32 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish32 - start32).count();
        auto start4 = std::chrono::high_resolution_clock::now();
        pos = net->predict_Double(search_key_double) * net->y_gap;
        auto finish4 = std::chrono::high_resolution_clock::now();
        // std::cout<< "search prediction 2------4------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish4 - start4).count() << std::endl;
        exp_recorder.prediction_time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish4 - start4).count();
        // auto start5 = std::chrono::high_resolution_clock::now();
        if (pos < 0)
        {
            pos = 0;
        }
        if (pos >= net->y_gap)
        {
            pos = net->y_gap - 1;
        }
        // std::cout << "pos: " << pos << std::endl;
        begin = pos > net->min_error ? pos - net->min_error + net->start_y : net->start_y;
        end = (pos + net->max_error + net->start_y) >= N ? N - 1 : (pos + net->start_y + net->max_error);

        auto finish5 = std::chrono::high_resolution_clock::now();
        // std::cout<< "return------5------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish5 - start5).count() << std::endl;
        // time_cost_5 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish5 - start5).count();
        // auto end = chrono::high_resolution_clock::now();
        // cout<< "predict time: " << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << endl;
        // std::cout<< "net->start_y: " << net->start_y << std::endl;
        // std::cout<< "net->y_gap: " << net->y_gap << std::endl;
        exp_recorder.time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish5 - start1).count();
        return pos + net->start_y;
    }

    long long search(uint64_t search_key, uint64_t &begin, uint64_t &end)
    {
        // auto start1 = std::chrono::high_resolution_clock::now();
        std::shared_ptr<Net> top = index[0][0];
        double search_key_double = double(search_key - top->start_x) / top->x_gap;
        // std::cout << "search_key: " << search_key << std::endl;
        // std::cout << "top->start_x: " << top->start_x << std::endl;
        // std::cout << "top->x_gap: " << top->x_gap << std::endl;
        // std::cout << "search_key_double: " << search_key_double << std::endl;
        // auto finish1 = std::chrono::high_resolution_clock::now();
        // std::cout<< "search : normalize 1------1------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish1 - start1).count() << std::endl;
        // time_cost_1 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish1 - start1).count();
        // auto start2 = std::chrono::high_resolution_clock::now();
        long long pos = top->predict_Double(search_key_double) * branch;
        // std::cout << "pos: " << pos << std::endl;
        // auto finish2 = std::chrono::high_resolution_clock::now();
        // std::cout<< "search prediction 1------2------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish2 - start2).count() << std::endl;
        // time_cost_2 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish2 - start2).count();
        if (pos < 0)
        {
            pos = 0;
        }
        else if (pos >= branch)
        {
            pos = branch - 1;
        }
        // auto start31 = std::chrono::high_resolution_clock::now();
        // std::shared_ptr<Net> net = index[1][pos];
        std::shared_ptr<Net> net = layer2_index[pos];
        // auto finish31 = std::chrono::high_resolution_clock::now();
        // std::cout<< "get model in layer2------31------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish31 - start31).count() << std::endl;
        // time_cost_31 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish31 - start31).count();
        // std::cout<< "pos: " << pos << std::endl;
        // std::cout<< "net->start_y: " << net->start_y << std::endl;
        // std::cout<< "search_key: " << search_key << std::endl;
        // auto start32 = std::chrono::high_resolution_clock::now();
        search_key_double = double(search_key - net->start_x) / net->x_gap;
        // auto finish32 = std::chrono::high_resolution_clock::now();
        // std::cout<< "search : normalize 2------32------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish32 - start32).count() << std::endl;
        // time_cost_32 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish32 - start32).count();
        // auto start4 = std::chrono::high_resolution_clock::now();
        pos = net->predict_Double(search_key_double) * net->y_gap;
        // auto finish4 = std::chrono::high_resolution_clock::now();
        // std::cout<< "search prediction 2------4------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish4 - start4).count() << std::endl;
        // time_cost_4 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish4 - start4).count();
        // auto start5 = std::chrono::high_resolution_clock::now();
        if (pos < 0)
        {
            pos = 0;
        }
        if (pos >= net->y_gap)
        {
            pos = net->y_gap - 1;
        }
        // std::cout << "pos: " << pos << std::endl;
        begin = pos > net->min_error ? pos - net->min_error + net->start_y : net->start_y;
        end = (pos + net->max_error + net->start_y) >= N ? N - 1 : (pos + net->start_y + net->max_error);

        // auto finish5 = std::chrono::high_resolution_clock::now();
        // std::cout<< "return------5------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish5 - start5).count() << std::endl;
        // time_cost_5 += std::chrono::duration_cast<std::chrono::nanoseconds>(finish5 - start5).count();
        // auto end = chrono::high_resolution_clock::now();
        // cout<< "predict time: " << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << endl;
        // std::cout<< "net->start_y: " << net->start_y << std::endl;
        // std::cout<< "net->y_gap: " << net->y_gap << std::endl;
        return pos + net->start_y;
    }

    void build_one_layer(ExpRecorder &exp_recorder, std::vector<uint64_t> data)
    {
        auto start = std::chrono::high_resolution_clock::now();
        N = data.size();
        records = data;
        uint64_t start_x = data[0];
        uint64_t end_x = data[N - 1];
        uint64_t key_gap = end_x - start_x;

        auto net = std::make_shared<Net>(start_x, key_gap, 0, N);
        if (is_model_reuse)
        {
            Histogram histogram(data);
            // Histogram histogram(data.begin(), data.size());
            string model_path;
            double distance;
            if (net->is_model_reusable(histogram, 1.0, model_path, distance))
            {
                // std::cout<< "model_path: " << model_path << std::endl;
                torch::load(net, (PATH + model_path + ".pt"));
                net->getParameters_Double();
            }
            // net->reuse_model(net, data);
        }
        else
        {
            string path = "./torch_models/" + tag + "/RMI_NN/" + to_string(1) + "/" + to_string(0) + "_" + to_string(0) + ".pt";
            std::ifstream fin(path);

            if (Constants::IS_RECORD_BUILD_TIME || !fin)
            {
                // std::cout<< "train model: " << path << std::endl;
                net->trainModel(data);
                torch::save(net, path);
            }
            else
            {
                torch::load(net, path);
                net->getParameters_Double();
            }
        }

        net->cal_errors(data);
        net->printParameters_Double();
        net->print_model_info();

        layer1_index.push_back(net);
        index.push_back(layer1_index);
        auto end = std::chrono::high_resolution_clock::now();
        exp_recorder.build_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    void build_two_layer(ExpRecorder &exp_recorder, std::vector<uint64_t> data)
    {
        auto start = std::chrono::high_resolution_clock::now();
        N = data.size();
        uint64_t start_x = data[0];
        uint64_t end_x = data[N - 1];
        uint64_t key_gap = end_x - start_x;

        auto net = std::make_shared<Net>(start_x, key_gap, 0, N);
        double distance;
        if (is_model_reuse)
        {
            Histogram histogram(data);
            // Histogram histogram(data.begin(), data.size());
            string model_path;
            if (net->is_model_reusable(histogram, 1.0, model_path, distance))
            {
                // reused_models.insert(model_path);
                // torch::load(net, (PATH + model_path + ".pt"));
                // net = net->models[model_path];
                // net->start_x = start_x;
                // net->x_gap = key_gap;
                // net->start_y = 0;
                // net->y_gap = N;
                // net->getParameters_Double();
                // torch::load(net, (PATH + model_path + ".pt"));
                // net->getParameters_Double();
                std::shared_ptr<Net> temp_net = net->models[model_path];
                net->w1_D = temp_net->w1_D;
                net->w2_D = temp_net->w2_D;
                net->b1_D = temp_net->b1_D;
                net->b2_D = temp_net->b2_D;
            }
            // net->reuse_model(net, data);
        }
        else
        {
            string path = "./torch_models/" + tag + "/RMI_NN/" + to_string(branch) + "/" + to_string(0) + "_" + to_string(0) + ".pt";
            std::ifstream fin(path);
            if (Constants::IS_RECORD_BUILD_TIME || !fin)
            {
                // std::cout<< "trainModel: " << path << std::endl;
                net->trainModel(data);
                torch::save(net, path);
            }
            else
            {
                // std::cout<< "load model: " << path << std::endl;
                torch::load(net, path);
                // std::cout<< "load finish: " << std::endl;
                net->getParameters_Double();
            }
        }
        // net->print_model_info();

        layer1_index.push_back(net);
        index.push_back(layer1_index);

        for (int k = 0; k < branch; k++)
        {
            std::vector<uint64_t> stage_temp_entites;
            stage2.push_back(stage_temp_entites);
        }
        for (uint64_t entity : data)
        {
            double res = net->predict_Double((entity - start_x) * 1.0 / key_gap);
            // std::cout<< "res: " << res << std::endl;
            long long pos = (long long)(res * branch);
            if (pos < 0)
            {
                pos = 0;
            }
            if (pos >= branch)
            {
                pos = branch - 1;
            }
            // if (entity > data[199110673] && pos < 98)
            // {
            //     std::cout<< "pos: " << pos << std::endl;
            // }
            // break;
            stage2[pos].push_back(entity);
        }

        std::vector<uint64_t> start_ys;
        uint64_t size_sum = 0;
        for (int i = 0; i < branch; i++)
        {
            start_ys.push_back(size_sum);
            // if(stage2[i].size() != 0 && i < 5500)
            //     std::cout<< "i: " << i << " stage2[i].size(): " << stage2[i].size() << " size_sum: " << size_sum << std::endl;
            size_sum += stage2[i].size();
        }

        uint64_t sub_start_x;
        uint64_t sub_key_gap;
        uint64_t start_ys_y;
        for (int j = 0; j < branch; j++)
        {
            // std::cout<< "size: " <<j << "th " << stage2[j].size() << std::endl;
            uint64_t sub_data_size = stage2[j].size();
            if (sub_data_size == 0)
            {
                auto net = std::make_shared<Net>(sub_start_x, sub_key_gap, start_ys_y, 1);
                net->getParameters_Double();
                layer2_index.push_back(net);
                net->insert_number_bound = 0;
                continue;
            }
            sub_start_x = stage2[j][0];
            sub_key_gap = stage2[j][sub_data_size - 1] - sub_start_x;
            start_ys_y = start_ys[j];
            auto net = std::make_shared<Net>(sub_start_x, sub_key_gap, start_ys_y, sub_data_size);

            // && sub_data_size >= Constants::BIN_NUM
            if (is_model_reuse)
            {
                Histogram histogram(stage2[j]);
                // Histogram histogram(stage2[j].begin(), stage2[j].size());
                string model_path;
                if (net->is_model_reusable(histogram, 1.0, model_path, distance))
                {
                    reused_models.insert(model_path);
                    torch::load(net, (PATH + model_path + ".pt"));
                    // net = net->models[model_path];
                    // net->start_x = sub_start_x;
                    // net->x_gap = sub_key_gap;
                    // net->start_y = start_ys[j];
                    // net->y_gap = sub_data_size;
                    net->getParameters_Double();
                }
            }
            else
            {
                string path = "./torch_models/" + tag + "/RMI_NN/" + to_string(branch) + "/" + to_string(1) + "_" + to_string(j) + ".pt";
                std::ifstream fin(path);
                if (Constants::IS_RECORD_BUILD_TIME || !fin)
                {
                    net->trainModel(stage2[j]);
                    torch::save(net, path);
                }
                else
                {
                    // std::cout<< "level 2 model_path: " << path << std::endl;
                    torch::load(net, path);
                    net->getParameters_Double();
                }
            }
            double gap = Constants::THRESHOLD - distance;
            gap = gap > 0 ? gap : 0.1;
            net->insert_number_bound = gap * sub_data_size / (1 - gap);
            net->cal_errors(stage2[j]);
            layer2_index.push_back(net);
            net->records = stage2[j];
            // break;
        }
        index.push_back(layer2_index);
        auto end = std::chrono::high_resolution_clock::now();
        exp_recorder.build_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    void insert_one_layer(uint64_t insert_data)
    {
    }

    void insert_two_layer(uint64_t insert_data, std::vector<uint64_t>::iterator begin, uint64_t inserted_num)
    {
        // TODO prediction
        std::shared_ptr<Net> top = index[0][0];
        double search_key_double = double(insert_data - top->start_x) / top->x_gap;

        long long pos = top->predict_Double(search_key_double) * branch;
        // std::cout << "search_key_double " << search_key_double <<std::endl;
        // std::cout << "pos: " << pos <<std::endl;
        if (pos < 0)
        {
            pos = 0;
        }
        else if (pos >= branch)
        {
            pos = branch - 1;
        }
        std::shared_ptr<Net> net = layer2_index[pos];
        net->insert_number++;
        // std::cout << "net->insert_number: " << net->insert_number << std::endl;
        // std::cout << "net->insert_number_bound: " << net->insert_number_bound << std::endl;
        if (net->insert_number > net->insert_number_bound)
        {
            // TODO rebuild!!!
            // auto start3 = std::chrono::high_resolution_clock::now();
            stage2[pos].push_back(insert_data);
            layer2_index[pos] = update_rebuild(stage2[pos], net->start_y);
            // auto end3 = std::chrono::high_resolution_clock::now();
            // std::cout<< "rebuild time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count() << std::endl;
        }
        else
        {
            // std::cout << "-------1------" <<std::endl;
            search_key_double = double(insert_data - net->start_x) / net->x_gap;
            pos = net->predict_Double(search_key_double) * net->y_gap + net->start_y;
            //TODO calibration
            uint64_t start = net->start_y;
            uint64_t end = net->start_y + net->y_gap + inserted_num;
            uint64_t mid = (start + end) / 2;

            // if (inserted_num > 196188009)
            // {
            //     std::cout << "inserted_num: ? " << inserted_num << std::endl;
            //     std::cout << "start: " << start <<std::endl;
            //     std::cout << "end: " << end <<std::endl;
            //     std::cout << "net->start_x: " << net->start_x <<std::endl;
            //     std::cout << "*(begin + net->start_y) : " << *(begin + net->start_y)  <<std::endl;
            // }
            // std::cout << "start: " << start <<std::endl;
            // std::cout << "end: " << end <<std::endl;
            // std::cout << "net->start_x: " << net->start_x <<std::endl;
            // std::cout << "*(begin + net->start_y) : " << *(begin + net->start_y)  <<std::endl;

            if (*(begin + net->start_y) != net->start_x)
            {
                uint64_t temp = *(begin + mid);
                while (temp != net->start_x && start <= end)
                {
                    if (temp < net->start_x)
                    {
                        start = mid + 1;
                    }
                    else if (temp > net->start_x)
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
                // net->start_y = mid;
            }
            //TODO insert

            start = net->start_y;
            end = net->start_y + net->y_gap;
            mid = (start + end) / 2;
            // std::cout<< "start: " << *(begin + start) << " end: " << *(begin + end) << " insert_data: " << insert_data << std::endl;
            while (start < end)
            {
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
        }
    }

    std::shared_ptr<Net> update_rebuild(std::vector<uint64_t> data, uint64_t start_y)
    {
        // std::cout << "update_rebuild: 1" << std::endl;
        // auto start = std::chrono::high_resolution_clock::now();
        uint64_t cardinality = data.size();
        uint64_t start_x = data[0];
        uint64_t key_gap = data[cardinality - 1] - start_x;
        key_gap = key_gap == 0 ? 1 : key_gap;
        auto net = std::make_shared<Net>(start_x, key_gap, start_y, cardinality);
        // auto finish = std::chrono::high_resolution_clock::now();
        // std::cout << "rebuild creation time1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        Histogram histogram(data);
        // finish = std::chrono::high_resolution_clock::now();
        // std::cout << "rebuild histogram time2: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << std::endl;
        string model_path;
        double distance;
        // std::cout << "update_rebuild: 2" << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        net->is_model_reusable(histogram, 1.0, model_path, distance);
        // finish = std::chrono::high_resolution_clock::now();
        // std::cout << "rebuild reuse time3: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        // torch::load(net, (PATH + model_path + ".pt"));
        // net->getParameters_Double();
        // TODO only use the parameter from models!!!
        // net = net->models[model_path];
        // double *w1_D;
        // double *w2_D;
        // double *b1_D;
        // double b2_D = 0.0;

        std::shared_ptr<Net> temp_net = net->models[model_path];

        net->w1_D = temp_net->w1_D;
        net->w2_D = temp_net->w2_D;
        net->b1_D = temp_net->b1_D;
        net->b2_D = temp_net->b2_D;

        net->start_x = start_x;
        net->x_gap = key_gap;
        net->start_y = start_y;
        net->y_gap = cardinality;
        net->records = data;
        // std::cout << "size 1: " << data.size() << std::endl;
        // std::cout << "size 2: " << net->models[model_path]->records.size() << std::endl;
        net->insert_number = 0;
        double gap = Constants::THRESHOLD - distance;
        gap = gap > 0 ? gap : 0.1;
        net->insert_number_bound = gap * cardinality / (1 - gap);
        net->min_error = cardinality * distance + (-net->model_infos[model_path].min_err * 100000) * 100000 / cardinality;
        net->max_error = cardinality * distance + (net->model_infos[model_path].max_err * 100000) * 100000 / cardinality;
        // std::cout << "update_rebuild: 4" << std::endl;
        // auto finish = std::chrono::high_resolution_clock::now();
        // std::cout << "rebuild time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << std::endl;
        // std::cout << "insert_number_bound: " << net->insert_number_bound << " min_error: " << net->min_error << " max_error: " << net->max_error << std::endl;
        return net;
    }

    void print_num()
    {
        uint64_t total_num = 0;
        for (size_t i = 0; i < layer2_index.size(); i++)
        {
            // std::shared_ptr<Net> net = layer2_index[i];
            total_num += layer2_index[i]->records.size();
        }
        std::cout << "total_num: " << total_num << std::endl;
    }

    void insert_two_layer_2(ExpRecorder &exp_recorder, uint64_t insert_data, uint64_t inserted_num)
    {
        // auto start1 = std::chrono::high_resolution_clock::now();
        std::shared_ptr<Net> top = index[0][0];
        double search_key_double = double(insert_data - top->start_x) / top->x_gap;
        long long pos = top->predict_Double(search_key_double) * branch;
        if (pos < 0)
        {
            pos = 0;
        }
        else if (pos >= branch)
        {
            pos = branch - 1;
        }
        std::shared_ptr<Net> net = layer2_index[pos];
        net->insert_number++;
        // auto end1 = std::chrono::high_resolution_clock::now();
        // exp_recorder.prediction_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count();
        // auto start4 = std::chrono::high_resolution_clock::now();
        uint64_t inserted_pos = 0;
        if (net->records[0] > insert_data)
        {
            // auto start3 = std::chrono::high_resolution_clock::now();
            net->records.insert(net->records.begin(), insert_data);
            // auto end3 = std::chrono::high_resolution_clock::now();
            // exp_recorder.insert_time_inner += std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count();
        }
        else if (net->records[net->records.size() - 1] < insert_data)
        {
            // auto start3 = std::chrono::high_resolution_clock::now();
            net->records.push_back(insert_data);
            inserted_pos = net->records.size();
            // auto end3 = std::chrono::high_resolution_clock::now();
            // exp_recorder.insert_time_inner += std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count();
        }
        else
        {
            long start = 0;
            long finish = net->records.size() - 1;
            long mid = 0;
            // std::vector<uint64_t> temp_points = net->records;
            while (start <= finish)
            {
                mid = (start + finish) / 2;
                uint64_t temp_0 = net->records[mid];
                uint64_t temp_1 = net->records[mid + 1];
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
                    // auto start3 = std::chrono::high_resolution_clock::now();
                    net->records.insert(net->records.begin() + mid + 1, insert_data);
                    // net->records.emplace(net->records.begin() + mid + 1, insert_data);
                    // auto end3 = std::chrono::high_resolution_clock::now();
                    // exp_recorder.insert_time_inner += std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count();
                    inserted_pos = mid + 1;
                    break;
                }
            }

            // std::cout<< "insert time 3 : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count() << std::endl;
        }
        // auto end4 = std::chrono::high_resolution_clock::now();
        // exp_recorder.insert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - start4).count();

        // std::cout<< "insert time 4 : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() << std::endl;
        // if((pos_l2 > inserted_pos && pos_l2 - inserted_pos <= net->min_error + net->insert_number_bound)
        // || (pos_l2 < inserted_pos && inserted_pos - pos_l2 <= net->max_error + net->insert_number_bound))
        // {
        //     return;
        // }
        // else
        // {
        //     layer2_index[pos] = update_rebuild_2(stage2[pos], net->start_y);
        // }
        // }
        if (net->insert_number > net->insert_number_bound)
        {
            auto start2 = std::chrono::high_resolution_clock::now();
            // net->records.push_back(insert_data);
            exp_recorder.rebuild_num++;
            layer2_index[pos] = update_rebuild(net->records, net->start_y);
            auto end2 = std::chrono::high_resolution_clock::now();
            exp_recorder.rebuild_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
            // std::cout<< "rebuild time------------: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count() << std::endl;
        }
    }

    bool search_after_insertion(ExpRecorder &exp_recorder, uint64_t search_key)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        std::shared_ptr<Net> top = index[0][0];
        double search_key_double = double(search_key - top->start_x) / top->x_gap;
        auto finish1 = std::chrono::high_resolution_clock::now();
        exp_recorder.vector_visit_time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish1 - start1).count();
        auto start2 = std::chrono::high_resolution_clock::now();
        long long pos = top->predict_Double(search_key_double) * branch;
        auto finish2 = std::chrono::high_resolution_clock::now();
        exp_recorder.prediction_time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish2 - start2).count();
        if (pos < 0)
        {
            pos = 0;
        }
        else if (pos >= branch)
        {
            pos = branch - 1;
        }
        auto start31 = std::chrono::high_resolution_clock::now();
        std::shared_ptr<Net> net = layer2_index[pos];

        auto finish31 = std::chrono::high_resolution_clock::now();
        exp_recorder.vector_visit_time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish31 - start31).count();

        search_key_double = double(search_key - net->start_x) / net->x_gap;
        auto start4 = std::chrono::high_resolution_clock::now();
        pos = net->predict_Double(search_key_double) * net->y_gap;
        auto finish4 = std::chrono::high_resolution_clock::now();
        exp_recorder.prediction_time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish4 - start4).count();
        auto start5 = std::chrono::high_resolution_clock::now();

        if (pos < 0)
        {
            pos = 0;
        }
        if (pos >= net->y_gap)
        {
            pos = net->y_gap - 1;
        }
        // std::cout << "pos: " << pos << std::endl;
        // uint64_t begin = pos > net->min_error ? pos - net->min_error + net->start_y : net->start_y;
        // uint64_t end = (pos + net->max_error + net->start_y) >= N ? N - 1 : (pos + net->start_y + net->max_error);

        long start = 0;
        long finish = net->records.size() - 1;
        long mid = 0;
        std::vector<uint64_t> temp_points = net->records;

        while (start <= finish)
        {
            mid = (start + finish) / 2;
            uint64_t temp = temp_points[mid];
            if (temp == search_key)
            {
                auto finish5 = std::chrono::high_resolution_clock::now();
                exp_recorder.time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish5 - start5).count();

                return true;
            }
            else if (temp < search_key)
            {
                start = mid + 1;
            }
            else
            {
                finish = mid - 1;
            }
        }
        auto finish5 = std::chrono::high_resolution_clock::now();
        exp_recorder.time += std::chrono::duration_cast<std::chrono::nanoseconds>(finish5 - start5).count();
        return false;
    }
};

#endif