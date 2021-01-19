#ifndef HISTOGRAM_CPP
#define HISTOGRAM_CPP

#include <vector>
// #include <string.h>
#include "../utils/Constants.h"
// using namespace std;
class Histogram
{
private:
    // long long binary_search(uint64_t);
    std::vector<uint64_t> entities;

    std::vector<uint64_t>::iterator begin;
    uint64_t data_size;

public:
    std::vector<double> hist;
    // Histogram(std::vector<uint64_t>);
    // double cal_dist(std::vector<double>);
    double cal_dist(std::vector<double> source_hist)
    {
        double dist_hist = std::max(source_hist[0], hist[0]);
        double temp_sum = 0.0;

        for (size_t i = 0; i < source_hist.size() - 1; i++)
        {
            temp_sum += source_hist[i] - hist[i];
            double temp = 0.0;
            if (source_hist[i + 1] + temp_sum > hist[i + 1] - temp_sum)
            {
                temp = source_hist[i + 1] + temp_sum;
            }
            else
            {
                temp = hist[i + 1] - temp_sum;
            }
            if (dist_hist < temp)
            {
                dist_hist = temp;
            }
        }
        return dist_hist;
    }

    Histogram(std::vector<uint64_t> data)
    {

        this->entities = data;
        uint64_t N = data.size();
        uint64_t start_x = data[0];
        uint64_t end_x = data[N - 1];

        double key_gap = (end_x - start_x) * 1.0 / Constants::BIN_NUM;
        double old_freq = 0.0;
        if (N < Constants::BIN_NUM)
        {
            for (size_t i = 1; i < Constants::BIN_NUM; i++)
            {
                int num = 0;
                for (size_t j = 0; j < N; j++)
                {
                    if (data[j] > start_x + key_gap * i && data[j] <= start_x + key_gap * (i + 1))
                    {
                        num++;
                    }
                    if (data[j] > start_x + key_gap * (i + 1))
                    {
                        break;
                    }
                    
                }
                hist.push_back(double(num) / N);
            }
        }
        else
        {
            // cout<< "key_gap: " << key_gap << endl;
            for (size_t i = 1; i < Constants::BIN_NUM; i++)
            {
                long long index = binary_search(start_x + key_gap * i);
                double freq = index * 1.0 / N - old_freq;
                hist.push_back(freq);
                old_freq = index * 1.0 / N;
                // cout<< "i: " << i << " freq:" << freq << endl;
            }
            // cout<< "finish:" << endl;
            hist.push_back(1 - old_freq);
        }

    }

    // Histogram(std::vector<uint64_t>::iterator begin, uint64_t data_size)
    // {

    //     this->begin = begin;
    //     this->data_size = data_size;
    //     uint64_t N = data_size;
    //     uint64_t start_x = *begin;
    //     uint64_t end_x = *(begin + N - 1);

    //     double key_gap = (end_x - start_x) * 1.0 / Constants::BIN_NUM;
    //     double old_freq = 0.0;
    //     if (N < Constants::BIN_NUM)
    //     {
    //         for (size_t i = 1; i < Constants::BIN_NUM; i++)
    //         {
    //             int num = 0;
    //             for (size_t j = 0; j < N; j++)
    //             {
    //                 if (*(begin+j) > start_x + key_gap * i && *(begin+j)  <= start_x + key_gap * (i + 1))
    //                 {
    //                     num++;
    //                 }
    //                 if (*(begin+j)  > start_x + key_gap * (i + 1))
    //                 {
    //                     break;
    //                 }
                    
    //             }
    //             hist.push_back(double(num) / N);
    //         }
    //     }
    //     else
    //     {
    //         // cout<< "key_gap: " << key_gap << endl;
    //         for (size_t i = 1; i < Constants::BIN_NUM; i++)
    //         {
    //             long long index = binary_search(start_x + key_gap * i);
    //             double freq = index * 1.0 / N - old_freq;
    //             hist.push_back(freq);
    //             old_freq = index * 1.0 / N;
    //             // cout<< "i: " << i << " freq:" << freq << endl;
    //         }
    //         // cout<< "finish:" << endl;
    //         hist.push_back(1 - old_freq);
    //     }

    // }

    long long binary_search(uint64_t key)
    {
        long long begin = 0;
        long long end = entities.size() - 1;
        long long mid = (begin + end) / 2;
        while(entities[mid] != key) {
            if (entities[mid] < key) 
            {
                if (entities[mid + 1] >= key) 
                {
                    break;
                }
                begin = mid;
            } 
            else if (entities[mid] > key)
            {
                if (entities[mid - 1] <= key)
                {
                    mid -= 1;
                    break;
                }
                end = mid;
            }
            mid = (begin + end) / 2;
        }
        return mid;
    }
    // long long binary_search(uint64_t key)
    // {
    //     long long start = 0;
    //     long long end = entities.size() - 1;
    //     long long mid = (start + end) / 2;
    //     while(*(begin+mid) != key) {
    //         if (*(begin+mid) < key) 
    //         {
    //             if (*(begin+mid+1) >= key) 
    //             {
    //                 break;
    //             }
    //             start = mid;
    //         } 
    //         else if (*(begin+mid) > key)
    //         {
    //             if (*(begin+mid-1) <= key)
    //             {
    //                 mid -= 1;
    //                 break;
    //             }
    //             end = mid;
    //         }
    //         mid = (start + end) / 2;
    //     }
    //     return mid;
    // }
};

#endif