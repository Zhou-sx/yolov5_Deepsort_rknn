#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <opencv2/opencv.hpp>
#include "featuretensor.h"
#include "tracker.h"
#include "datatype.h"
#include "model.hpp"
#include <vector>

using std::vector;

class DeepSort {
public:    
    DeepSort(std::string modelPath, int batchSize, int featureDim, int cpu_id, rknn_core_mask npu_id);
    ~DeepSort();

public:
    void sort(cv::Mat& frame, vector<DetectBox>& dets);
    void sort_interval(cv::Mat& frame, vector<DetectBox>& dets);
    int  track_process();
    void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes);

private:
    void sort(cv::Mat& frame, DETECTIONS& detections);
    void sort(cv::Mat& frame, DETECTIONSV2& detectionsv2);   
    void init();

private:
    std::string enginePath;
    int batchSize;
    int featureDim;
    cv::Size imgShape;
    float confThres;
    float nmsThres;
    int maxBudget;
    float maxCosineDist;

    const int track_interval = 1; 
private:
    vector<RESULT_DATA> result;
    vector<std::pair<CLSCONF, DETECTBOX>> results;
    tracker* objTracker;
    FeatureTensor* featureExtractor1;
    FeatureTensor* featureExtractor2;
    rknn_core_mask npu_id;
    int cpu_id;
};

#endif  //deepsort.h
