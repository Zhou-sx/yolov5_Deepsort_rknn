#ifndef FEATURETENSOR_H
#define FEATURETENSOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "model.hpp"
#include "datatype.h"
#include "rknn_fp.h"
#include "resize.h"

using std::vector;

class FeatureTensor :public rknn_fp{
public:
    using rknn_fp::rknn_fp;
    void init(cv::Size, int, int);
    bool getRectsFeature(const cv::Mat& img, DETECTIONS& det);
    void doInference(vector<cv::Mat>& imgMats, DETECTIONS& det);

public:
    cv::Size imgShape;
    int featureDim;
    PreResize pre_do;
};

#endif
