#include "featuretensor.h"

void FeatureTensor::doInference(vector<cv::Mat>& imgMats, DETECTIONS& det) {
    for (int i = 0;i < imgMats.size();i++){
		inference(imgMats[i].data);
		float* output = (float *)_output_buff[0];
		for (int j = 0; j < featureDim; ++j)
            det[i].feature[j] = output[j];
	}
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& det) {
    std::vector<cv::Mat> mats;
    for (auto& dbox : det) {
        cv::Rect rect = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                                 int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rect.x -= (rect.height * 0.5 - rect.width) * 0.5;
        rect.width = rect.height * 0.5;
        rect.x = (rect.x >= 0 ? rect.x : 0);
        rect.y = (rect.y >= 0 ? rect.y : 0);
        rect.width = (rect.x + rect.width <= img.cols ? rect.width : (img.cols - rect.x));
        rect.height = (rect.y + rect.height <= img.rows ? rect.height : (img.rows - rect.y));
        cv::Mat tempMat = img(rect).clone();
        cv::resize(tempMat, tempMat, imgShape);
        mats.push_back(tempMat);
    }
    doInference(mats, det);
    return true;
}