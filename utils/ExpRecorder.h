#ifndef EXPRECORDER_H
#define EXPRECORDER_H
#include <iostream>
#include <string.h>

class ExpRecorder
{
public:
    long time = 0;
    long build_time = 0;
    long prediction_time = 0;
    long insert_time = 0;
    long insert_time_inner = 0;
    long search_time = 0;
    long vector_visit_time = 0;
    long rebuild_time = 0;
    long rebuild_num = 0;
    ExpRecorder()
    {
    }

    void clear()
    {
        time = 0;
        build_time = 0;
        prediction_time = 0;
        search_time = 0;
        vector_visit_time = 0;
        rebuild_time = 0;
        insert_time_inner = 0;
        rebuild_num = 0;
    }
};

#endif