#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <string>
class Constants
{
public:
    static const bool IS_RECORD_BUILD_TIME = false;
    static const bool MODEL_REUSE = false;
    static const int WIDTH = 4;
    static const int BIN_NUM = 10;
    static const int EPOCH = 500;
    static const constexpr double THRESHOLD = 0.1;

    static const int BATCH_THRESHOLD = 1000000;
    static const int BATCH_SIZE = 200000;

    Constants();
};

#endif