#ifndef BOX_H
#define BOX_H
#include <vector>
#include "opencv2/opencv.hpp"

// 标签
#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64

typedef struct DetectBox {
    DetectBox(float x1=0, float y1=0, float x2=0, float y2=0, 
            float confidence=0, float classID=-1, float trackID=-1) {
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
        this->confidence = confidence;
        this->classID = classID;
        this->trackID = trackID;
    }
    char name[OBJ_NAME_MAX_SIZE];
    float x1, y1, x2, y2;
    float confidence;
    float classID;
    float trackID;
} DetectBox;
#endif