#ifndef MODELTOOLS_H
#define MODELTOOLS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>

#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

#include "Constants.h"
#include "../entities/Histogram.h"
#include <xmmintrin.h> //SSE指令集需包含词头文件
#include <immintrin.h>

#include <map>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <dirent.h>

using namespace at;
using namespace torch::nn;
using namespace torch::optim;

struct ModelInfo
{
    double min_err;
    double max_err;
};

struct Net : torch::nn::Module
{
public:
    
    string PATH = "./pre-train/trained_models/nn/4/";
    // string PATH = "../../pre-train/trained_models/nn/4/";

    float *w1;
    float *w2;
    float *b1;
    float b2 = 0.0;

    double *w1_D;
    double *w2_D;
    double *b1_D;
    double b2_D = 0.0;

    inline static std::map<string, std::vector<double>> model_features;
    inline static std::map<string, std::shared_ptr<Net>> models;
    inline static std::map<string, std::vector<double>> lr_models;
    inline static std::map<string, std::vector<double>> lr_model_features;

    inline static std::map<string, ModelInfo> model_infos;

    float gap;

    double slope;
    double intercept;

    bool is_linear;

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    uint64_t max_error = 0;
    uint64_t min_error = 0;

    uint64_t insert_number_bound = 0;

    uint64_t insert_number = 0;

    uint64_t start_x;
    uint64_t x_gap;
    uint64_t start_y;
    uint64_t y_gap;
    // uint64_t cardinality;
    // uint64_t end_y;

    Net(uint64_t start_x, uint64_t x_gap, uint64_t start_y, uint64_t y_gap, bool is_linear)
    {
        this->start_x = start_x;
        this->x_gap = x_gap;
        this->start_y = start_y;
        this->y_gap = y_gap;
        this->is_linear = is_linear;
    }

    Net(uint64_t start_x, uint64_t x_gap, uint64_t start_y, uint64_t y_gap)
    {
        this->start_x = start_x;
        this->x_gap = x_gap;
        this->start_y = start_y;
        this->y_gap = y_gap;

        // TODO .bias(false)
        fc1 = register_module("fc1", torch::nn::Linear(1, Constants::WIDTH));
        fc1->to(torch::kFloat64);
        fc2 = register_module("fc2", torch::nn::Linear(Constants::WIDTH, 1));
        fc2->to(torch::kFloat64);

        torch::nn::init::uniform_(fc1->weight, 0, 1.0 / Constants::WIDTH);
        torch::nn::init::uniform_(fc1->bias, 0, 1.0 / Constants::WIDTH);
        torch::nn::init::uniform_(fc2->weight, 0, 1.0 / Constants::WIDTH);
        // torch::nn::init::uniform_(fc2->bias, 0, 1.0 / Constants::WIDTH);

        // size_t size_of_float = sizeof(float);
        size_t size_of_double = sizeof(double);

        // w1 = (float *)_mm_malloc(width * size_of_float, 8 * size_of_float);
        // w2 = (float *)_mm_malloc(width * size_of_float, 8 * size_of_float);
        // b1 = (float *)_mm_malloc(width * size_of_float, 8 * size_of_float);

        w1_D = (double *)_mm_malloc(Constants::WIDTH * size_of_double, 8 * size_of_double);
        w2_D = (double *)_mm_malloc(Constants::WIDTH * size_of_double, 8 * size_of_double);
        b1_D = (double *)_mm_malloc(Constants::WIDTH * size_of_double, 8 * size_of_double);
    }

    // Net(int inputNum, uint64_t start_x, uint64_t x_gap, uint64_t cardinality)
    // {
    //     // this->inputNum = inputNum;
    //     this->start_x = start_x;
    //     this->x_gap = x_gap;
    //     this->cardinality = cardinality;

    //     // TODO .bias(false)
    //     fc1 = register_module("fc1", torch::nn::Linear(inputNum, Constants::WIDTH));
    //     fc1->to(torch::kFloat64);
    //     fc2 = register_module("fc2", torch::nn::Linear(Constants::WIDTH, 1));
    //     fc2->to(torch::kFloat64);

    //     torch::nn::init::uniform_(fc1->weight, 0, 1.0);
    //     torch::nn::init::uniform_(fc1->bias, 0, 1.0);
    //     torch::nn::init::uniform_(fc2->weight, 0, 1.0);
    //     torch::nn::init::uniform_(fc2->bias, 0, 1.0);

    //     // size_t size_of_float = sizeof(float);
    //     size_t size_of_double = sizeof(double);

    //     // w1 = (float *)_mm_malloc(width * size_of_float, 8 * size_of_float);
    //     // w2 = (float *)_mm_malloc(width * size_of_float, 8 * size_of_float);
    //     // b1 = (float *)_mm_malloc(width * size_of_float, 8 * size_of_float);

    //     w1_D = (double *)_mm_malloc(Constants::WIDTH * size_of_double, 8 * size_of_double);
    //     w2_D = (double *)_mm_malloc(Constants::WIDTH * size_of_double, 8 * size_of_double);
    //     b1_D = (double *)_mm_malloc(Constants::WIDTH * size_of_double, 8 * size_of_double);
    // }

    // void reuse(string model_path)
    // {
    //     std::shared_ptr<Net> net = models[model_path];
    //     // std::cout<<"net print" << std::endl;
    //     // net->printParameters_Double();
    //     memcpy(w1_D, net->w1_D, sizeof(net->w1_D));
    //     memcpy(b1_D, net->b1_D, sizeof(net->b1_D));
    //     memcpy(w2_D, net->w2_D, sizeof(net->w2_D));
    //     b2_D = net->b2_D;
    //     // std::cout<<"this print" << std::endl;
    //     // this->printParameters_Double();
    // }

    void getParameters_Double()
    {
        torch::Tensor p1 = this->parameters()[0];
        torch::Tensor p2 = this->parameters()[1];
        torch::Tensor p3 = this->parameters()[2];
        torch::Tensor p4 = this->parameters()[3];
        p1 = p1.reshape({Constants::WIDTH, 1});
        for (size_t i = 0; i < Constants::WIDTH; i++)
        {
            w1_D[i] = p1.select(0, i).item().toDouble();
        }

        p2 = p2.reshape({Constants::WIDTH, 1});
        for (size_t i = 0; i < Constants::WIDTH; i++)
        {
            b1_D[i] = p2.select(0, i).item().toDouble();
        }

        p3 = p3.reshape({Constants::WIDTH, 1});
        for (size_t i = 0; i < Constants::WIDTH; i++)
        {
            w2_D[i] = p3.select(0, i).item().toDouble();
        }

        b2_D = p4.item().toDouble();
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // cout<< "x0: " << x << endl;
        return fc2->forward(torch::relu(fc1->forward(x)));
    }

    torch::Tensor predict(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    double predict_Double_LR(double key)
    {
        return key * slope + intercept;
    }

    double predict_Double(double key)
    {
        // cout<< "predict_Double: 1" << endl;
        int blocks = Constants::WIDTH / 4;
        int rem = Constants::WIDTH % 4;
        int move_back = blocks * 4;
        __m256d DLoad_w1, DLoad_b1, DLoad_w2;
        __m256d temp1, temp2, temp3;
        __m256d DSum0 = _mm256_setzero_pd();
        __m256d DLoad0_x, DLoad0_zeros;
        // _mm_load1_ps
        DLoad0_x = _mm256_set_pd(key, key, key, key);
        DLoad0_zeros = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        double result;
        size_t size_of_double = sizeof(double);
        for (int i = 0; i < blocks; i++)
        {
            // TODO change w1
            DLoad_w1 = _mm256_load_pd(w1_D);
            DLoad_b1 = _mm256_load_pd(b1_D);
            DLoad_w2 = _mm256_load_pd(w2_D);
            temp1 = _mm256_mul_pd(DLoad0_x, DLoad_w1);
            temp2 = _mm256_add_pd(temp1, DLoad_b1);
            temp1 = _mm256_max_pd(temp2, DLoad0_zeros);
            temp2 = _mm256_mul_pd(temp1, DLoad_w2);
            DSum0 = _mm256_add_pd(DSum0, temp2);

            w1_D += 4;
            b1_D += 4;
            w2_D += 4;
        }
        // result = fSum0[0];
        result = 0;
        if(blocks > 0)
        {
            result += DSum0[0] + DSum0[1] + DSum0[2] + DSum0[3];
        }
        for (size_t i = 0; i < rem; i++)
        {
            result += activation(key * w1_D[i] + b1_D[i]) * w2_D[i];
        }
        result += b2_D;
        w1_D -= move_back;
        b1_D -= move_back;
        w2_D -= move_back;
        // cout << result << endl;
        return result;
    }


    double activation(double val)
    {
        // return std::max(val, 0.0);
        if (val > 0.0)
        {
            return val;
        }
        return 0.0;
        // return 1.0 / (1 + exp(-val)); // sigmoid
    }

    // void reuse_model(auto net, std::vector<uint64_t> data)
    // {
    //     Histogram histogram(data);
    //     string model_path;
    //     double distance;
    //     if(is_model_reusable(histogram, 1.0, model_path, distance))
    //     {
    //         torch::load(net, (PATH + model_path + ".pt"));
    //         getParameters_Double();
    //     }
    // }

    void printParameters_Double()
    {
        for (size_t i = 0; i < Constants::WIDTH; i++)
        {
            std::cout<< w1_D[i] << " ";
        }
        std::cout<< std::endl;
        for (size_t i = 0; i < Constants::WIDTH; i++)
        {
            std::cout<< b1_D[i] << " ";
        }
        std::cout<< std::endl;
        for (size_t i = 0; i < Constants::WIDTH; i++)
        {
            std::cout<< w2_D[i] << " ";
        }
        std::cout<< std::endl;
        std::cout<< b2_D << std::endl;
    }

    void trainModel(std::vector<uint64_t> data)
    {
        // this->start_y = start_y;
        // this->end_y = start_y + data.size() - 1;
        uint64_t N = data.size();
        // std::vector<uint64_t> index;
        // for (size_t i = 0; i < N; i++)
        // {
        //     index.push_back(i);
        // }
        if (N <= 800000000)
        {
            std::vector<double> keys;
            std::vector<double> ys;

            for (size_t i = 0; i < N; i++)
            {
                keys.push_back((data[i] - start_x) * 1.0 / x_gap);
                ys.push_back(i * 1.0 / N);
            }
            torch::Tensor x = torch::tensor(keys, at::kCUDA).reshape({(long long)N, 1});
            torch::Tensor y = torch::tensor(ys, at::kCUDA).reshape({(long long)N, 1});
            this->to(torch::kCUDA);
            torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(1e-3));
            if (N >= Constants::BATCH_THRESHOLD)
            {
                int batch_num = N / Constants::BATCH_SIZE;
                auto x_chunks = x.chunk(batch_num, 0);
                auto y_chunks = y.chunk(batch_num, 0);
                for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
                {
                    // std::cout<< "-----------epoch-------------: " << epoch << std::endl;
                    for (size_t i = 0; i < batch_num; i++)
                    {
                        optimizer.zero_grad();
                        torch::Tensor loss = torch::l1_loss(this->forward(x_chunks[i]), y_chunks[i]);
                        loss.to(torch::kCUDA);
                        loss.backward();
                        optimizer.step();
                        // std::cout<< "loss: " << loss << std::endl;
                    }
                }
            }
            else
            {
                for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
                {
                    optimizer.zero_grad();
                    torch::Tensor prediction = this->forward(x);
                    torch::Tensor loss = torch::l1_loss(prediction, y);
                    loss.to(torch::kCUDA);
                    loss.backward();
                    optimizer.step();
                    // std::cout<< "loss: " << loss << std::endl;
                }
            }

            // std::vector<double> keys;
            // std::vector<double> ys;

            // // auto rng = std::default_random_engine{};
            // // std::shuffle(std::begin(index), std::end(index), rng);
            // int gap = 1;
            // // TODO uncomment the below lines can accelerate
            // // if (N >= Constants::BATCH_THRESHOLD)
            // // {
            // //     gap = N / Constants::BATCH_THRESHOLD + 1;
            // // }
            // for (size_t i = 0; i < N; i=i+gap)
            // {
            //     keys.push_back((data[i] - start_x) * 1.0 / x_gap);
            //     ys.push_back(i * 1.0 / N);
            //     // if (N < 100000000)
            //     // {
            //     //     std::cout<< "x: " <<  (data[i] - start_x) * 1.0 / x_gap << "  y: " << i * 1.0 / N << std::endl;
            //     // }
            // }

            // // std::cout<< "keys.size(): " << keys.size() << std::endl;

            // torch::Tensor x = torch::tensor(keys, at::kCUDA).reshape({(long long)keys.size(), 1});
            // torch::Tensor y = torch::tensor(ys, at::kCUDA).reshape({(long long)keys.size(), 1});
            // // x = torch::nn::functional::normalize(x);
            // // torch::nn::functional::normalize(y);
            // // float gap = (ys[N - 1] - ys[0]) * 1.0  / ((keys[N - 1] - keys[0]) * 128);
            // this->to(torch::kCUDA);
            // torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(1e-3));
            // for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
            // {
            //     optimizer.zero_grad();
            //     torch::Tensor prediction = this->forward(x);
            //     torch::Tensor loss = torch::l1_loss(prediction, y);
            //     loss.to(torch::kCUDA);
            //     loss.backward();
            //     optimizer.step();
            //     // if (loss.item().toDouble() < 0.1)
            //     // {
            //         // std::cout<< "loss: " << loss << std::endl;
            //         // std::cout<< "prediction: " << prediction << std::endl;
            //     // }
            // }
        }
        // else
        // {
        //     // auto rng = std::default_random_engine{};
        //     // std::shuffle(std::begin(data), std::end(data), rng);
        //     std::vector<std::vector<double>> keys;
        //     std::vector<std::vector<double>> ys;
        //     for (size_t i = 0; i < 2; i++)
        //     {
        //         std::vector<double> temp_keys;
        //         keys.push_back(temp_keys);
        //         std::vector<double> temp_ys;
        //         ys.push_back(temp_ys);
        //     }

        //     for (size_t i = 0; i < N; i++)
        //     {
        //         keys[i%2].push_back((data[i] - start_x) * 1.0 / x_gap);
        //         ys[i%2].push_back((i + start_y) * 1.0 / cardinality);
        //     }

        //     this->to(torch::kCUDA);
        //     torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(1e-3 * this->gap));
        //     for (size_t j = 0; j < 1; j++)
        //     {   
        //         uint64_t half_N = keys[j].size();
        //         torch::Tensor x = torch::tensor(keys[j], at::kCUDA).reshape({(long long)half_N, 1});
        //         torch::Tensor y = torch::tensor(ys[j], at::kCUDA).reshape({(long long)half_N, 1});

        //         uint64_t batch_size = Constants::BATCH_SIZE;
        //         uint64_t batch_num = half_N / batch_size;
        //         auto x_chunks = x.chunk(batch_num, 0);
        //         auto y_chunks = y.chunk(batch_num, 0);
        //         for (size_t epoch = 0; epoch < Constants::EPOCH; epoch++)
        //         {
        //             for (size_t i = 0; i < batch_num; i++)
        //             {
        //                 optimizer.zero_grad();
        //                 torch::Tensor loss = torch::l1_loss(this->forward(x_chunks[i]), y_chunks[i]);
        //                 loss.to(torch::kCUDA);
        //                 loss.backward();
        //                 optimizer.step();
        //             }
        //         }
        //     }
        // }
        
        this->getParameters_Double();
    }

    void print_model_info()
    {
        std::cout<< "min_err: " << min_error << std::endl;
        std::cout<< "max_err: " << max_error << std::endl;
        std::cout<< "start_x: " << start_x << std::endl;
        std::cout<< "x_gap: " << x_gap << std::endl;
        std::cout<< "start_y: " << start_y << std::endl;
        std::cout<< "y_gap: " << y_gap << std::endl;
    }

    void cal_errors(std::vector<uint64_t> data)
    {
        uint64_t N = data.size();
        for (size_t i = 0; i < N; i++)
        {
            double res = predict_Double((data[i] - start_x) * 1.0 / x_gap);

            // std::cout << "x: " << (data[i] - start_x) * 1.0 / x_gap << " y: " << i * 1.0 / N  << " res: " << res << std::endl;
            
            long long pos = res * N;
            
            if (pos < 0)
            {
                pos = 0;
            }
            if (pos >= N)
            {
                pos = N - 1;
            }
            
            if (i > pos)
            {
                uint64_t error = i - pos;
                if (error > max_error)
                {
                    max_error = error;
                }
            }
            if (i < pos)
            {
                uint64_t error = pos - i;
                if (error > min_error)
                {
                    min_error = error;
                }
            }
        }
    }

    bool is_model_reusable(Histogram target_hist, double threshold, string& model_path, double& distance)
    {
        // load_trained_models(PATH);
        double min_dist = 1.0;
        std::map<string, std::vector<double>>::iterator iter;
        iter = model_features.begin();

        while (iter != model_features.end())
        {
            double temp_dist = target_hist.cal_dist(iter->second);
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                model_path = iter->first;
                if (min_dist <= Constants::THRESHOLD)
                {
                    break;
                }
            }
            iter++;
        }
        distance = min_dist;
        return true;
    }

    static bool is_model_lr_reusable(Histogram target_hist, double threshold, double& p_a, double& p_b, double& distance)
    {
        double min_dist = 1.0;
        std::map<string, std::vector<double>>::iterator iter;
        iter = lr_model_features.begin();
        string model_path;
        while (iter != lr_model_features.end())
        {
            // std::cout<< "iter->second: " << iter->second << std::endl;
            double temp_dist = target_hist.cal_dist(iter->second);
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                model_path = iter->first;
                if (min_dist <= Constants::THRESHOLD)
                {
                    break;
                }
            }
            iter++;
        }
        p_a = lr_models[model_path][0];
        p_b = lr_models[model_path][1];
        distance = min_dist;
        return true;
    }

    static void load_trained_lr_models(string ppath)
    {
        std::cout<< "load begin" << std::endl;
        auto load_start = std::chrono::high_resolution_clock::now();
        if(lr_model_features.size() > 0 and lr_model_features.size() > 0)
        {
            return;
        } 
        struct dirent *ptr;    
        DIR *dir;
        dir = opendir(ppath.c_str()); 
        std::vector<string> files;
        // string best_prefix;
        // double min_dist = 1.0;
        // Histogram histogram(entities, 100);
        while((ptr=readdir(dir))!=NULL)
        {
            if(ptr->d_name[0] == '.')
                continue;
            string path = ptr->d_name;
            int find_result = path.find(".json");
            if(find_result > 0 && find_result <= path.length())
            {
                std::ifstream read(ppath + path);
                json in = json::parse(read);
                string features = in["features"];
                string parameters = in["parameters"];
                // remove " "
                boost::trim_if(features, [](char c) { return std::ispunct(c); });

                std::vector<double> hist;
                std::vector<double> lr_parameters;
                std::string token;
                std::istringstream tokenStream(features);
                std::string lr_token;
                std::istringstream lr_tokenStream(parameters);
                while (std::getline(tokenStream, token, ','))
                {
                    hist.push_back(stod(token));
                }
                while (std::getline(lr_tokenStream, lr_token, ','))
                {
                    lr_parameters.push_back(stod(lr_token));
                }
                // std::cout<< "lr_parameters:" << lr_parameters << std::endl;
                string temp_prefix = path.substr(0, path.find(".json"));
                lr_model_features.insert(std::pair<string, std::vector<double>>(temp_prefix, hist));
                lr_models.insert(std::pair<string, std::vector<double>>(temp_prefix, lr_parameters));
            }
        }
        auto load_finish = std::chrono::high_resolution_clock::now();
        std::cout<< "load finish: " << std::chrono::duration_cast<std::chrono::nanoseconds>(load_finish - load_start).count() << std::endl;
    }

    static void load_trained_models(string ppath)
    {
        std::cout<< "load begin: " << ppath << std::endl;
        auto load_start = std::chrono::high_resolution_clock::now();
        if(model_features.size() > 0)
        {
            // cout<< "already loaded" << endl;
            return;
        } 
        struct dirent *ptr;    
        DIR *dir;
        dir = opendir(ppath.c_str()); 
        std::vector<string> files;
        // string best_prefix;
        // double min_dist = 1.0;
        // Histogram histogram(entities, 100);
        while((ptr=readdir(dir))!=NULL)
        {
            if(ptr->d_name[0] == '.')
                continue;
            string path = ptr->d_name;
            int find_result = path.find(".json");
            if(find_result > 0 && find_result <= path.length())
            {
                std::ifstream read(ppath + path);
                json in = json::parse(read);
                string features = in["features"];

                double min_err = in["min_err"];
                double max_err = in["max_err"];

                ModelInfo modelInfo;
                modelInfo.min_err = min_err;
                modelInfo.max_err = max_err;
                // remove " "
                boost::trim_if(features, [](char c) { return std::ispunct(c); });

                std::vector<double> hist;
                std::string token;
                std::istringstream tokenStream(features);
                while (std::getline(tokenStream, token, ','))
                {
                    hist.push_back(stod(token));
                }
                string temp_prefix = path.substr(0, path.find(".json"));
                model_features.insert(std::pair<string, std::vector<double>>(temp_prefix, hist));

                model_infos.insert(std::pair<string, ModelInfo>(temp_prefix, modelInfo));
            }
            find_result = path.find(".pt");
            if(find_result > 0 && find_result <= path.length())
            {

                std::shared_ptr<Net> net = std::make_shared<Net>(0, 1, 0, 1);
                // auto start = std::chrono::high_resolution_clock::now();
                torch::load(net, ppath + path);
                // auto end = std::chrono::high_resolution_clock::now();
                // std::cout<< "build time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;
                net->getParameters_Double();
                string temp_prefix = path.substr(0, path.find(".pt"));
                models.insert(std::pair<string, std::shared_ptr<Net>>(temp_prefix, net));
            }
            // cout<< "min_dist: " << min_dist << endl;
            // cout<< "best_prefix: " << best_prefix << endl;
        }
        // std::cout<< "load finish" << std::endl;
        
        std::map<string, std::vector<double>>::iterator iter;
        iter = model_features.begin();

        while (iter != model_features.end())
        {
            // cout<< "iter->first: " << iter->first << endl;
            iter++;
        }
        closedir(dir);

        auto load_finish = std::chrono::high_resolution_clock::now();
        std::cout<< "load finish: " << std::chrono::duration_cast<std::chrono::nanoseconds>(load_finish - load_start).count() << std::endl;
    }

};

#endif