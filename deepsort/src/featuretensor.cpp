#include <queue>
#include <iostream>

#include "featuretensor.h"
#include "mytime.h"


void FeatureTensor::init(cv::Size netShape, int featureDim, int channel){
    this->imgShape = netShape;
    this->featureDim = featureDim;
    this->pre_do = PreResize(netShape.height, netShape.width, channel);
}

void FeatureTensor::doInference(vector<cv::Mat>& imgMats, DETECTIONS& det) {
    std::queue<float> history_time;
	float sum_time = 0;
	int cost_time = 0; // rknn接口查询返回
	float npu_performance = 0.0;

    for (int i = 0;i < imgMats.size();i++){
        
		cost_time = inference(imgMats[i].data);
        // std::cout << "in deepsort doInference: " << i << "\n";
		float* output = (float *)_output_buff[0];
		for (int j = 0; j < featureDim; ++j)
            det[i].feature[j] = output[j];
        npu_performance = cal_NPU_performance(history_time, sum_time, cost_time / 1.0e3);
        // printf("Deepsort: %f NPU(%d) performance : %f\n", what_time_is_it_now()/1000, _cpu_id, npu_performance);
	}
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& det) {
    std::vector<cv::Mat> mats;
    
    double timeBeforeGetRectsFeature = what_time_is_it_now();

    for (auto& dbox : det) {
        cv::Rect rect = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                                 int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        // std::cout << img.cols << " " << img.rows << "\n";
        // std::cout << dbox.tlwh(0) << " " << dbox.tlwh(1) << " "  << dbox.tlwh(2) << " "  << dbox.tlwh(3) << "\n";

        rect.x -= (rect.height * 0.5 - rect.width) * 0.5;
        rect.width = rect.height * 0.5;
        rect.x = (rect.x >= 0 ? rect.x : 0);
        rect.y = (rect.y >= 0 ? rect.y : 0);
        rect.width = (rect.x + rect.width <= img.cols ? rect.width : (img.cols - rect.x));
        rect.height = (rect.y + rect.height <= img.rows ? rect.height : (img.rows - rect.y));

        if (rect.width < 0 || rect.height < 0) continue;
        // std::cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << "\n";
        cv::Mat tempMat = img(rect).clone();
        /*
        if (tempMat.rows < 128 || tempMat.cols < 256) {
            std::cout << tempMat.rows << " " << tempMat.cols << "\n";
        }
        */
        if (tempMat.size().empty()) {
            std::cout << "tempMat is empty: " << tempMat.cols << " " << tempMat.rows << "\n";
            continue;
        }
        cv::resize(tempMat, tempMat, imgShape);  // opencv
        // pre_do.resize(tempMat, tempMat);  // rga
        mats.push_back(tempMat);
    }
    
    doInference(mats, det);

    double timeAfterGetRectsFeature = what_time_is_it_now();
    std::cout << "--------Time cost in getRectsFeature: " << timeAfterGetRectsFeature- timeBeforeGetRectsFeature << "\n";

    // std::cout << "in deepsort inference: " << mats.size() << "\n";
    return true;
}