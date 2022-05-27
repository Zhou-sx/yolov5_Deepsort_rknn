#ifndef FEATURETENSOR_H
#define FEATURETENSOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "model.hpp"
#include "datatype.h"
#include "rknn_fp.h"

using std::vector;

class FeatureTensor :public rknn_fp{
public:
    using rknn_fp::rknn_fp;

    bool getRectsFeature(const cv::Mat& img, DETECTIONS& det);
    void doInference(vector<cv::Mat>& imgMats, DETECTIONS& det);

public:
    cv::Size imgShape;
    int featureDim;
};

#endif
