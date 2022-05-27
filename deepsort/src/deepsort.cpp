#include <unistd.h>
#include <string>
#include <iostream>

#include "deepsort.h"
#include "main.h"
using namespace std;

struct video_property;
extern queue<imageout_idx> queueDetOut;  // output queue
extern queue<imageout_idx> queueOutput;  		 // output queue 目标追踪输出队列
extern video_property video_probs;       // 视频属性类
extern bool bDetecting;                  // 目标检测进程状态
extern bool bTracking;                   // 目标追踪进程状态               
extern double end_time;                  // 整个视频追踪结束

DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim, int cpu_id, rknn_core_mask npu_id) {
    this->npu_id = npu_id;
    this->cpu_id = cpu_id;
    this->enginePath = modelPath;
    this->batchSize = batchSize;
    this->featureDim = featureDim;
    this->imgShape = cv::Size(128, 256);
    this->maxBudget = 100;
    this->maxCosineDist = 0.2;
    init();
}

void DeepSort::init() {
    objTracker = new tracker(maxCosineDist, maxBudget);
    featureExtractor = new FeatureTensor(enginePath.c_str(), cpu_id, npu_id, 1, 1);
    featureExtractor->imgShape = imgShape;
    featureExtractor->featureDim = featureDim;
}

DeepSort::~DeepSort() {
    delete objTracker;
}

void DeepSort::sort(cv::Mat& frame, vector<DetectBox>& dets) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    vector<CLSCONF> clsConf;
    
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.confidence));
    }
    result.clear();
    results.clear();
    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }
    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].confidence = c.conf;
    }
}


void DeepSort::sort(cv::Mat& frame, DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        //result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}

void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detectionsv2);
        result.clear();
        results.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh()));
        }
    }
}

int DeepSort::track_process(){
    while (1) 
	{
		if (queueDetOut.empty()) {
            if(!bDetecting){
                // queueDetOut为空并且 bDetecting标志已经全部检测完成 说明追踪完毕了
                end_time = what_time_is_it_now();
                bTracking = false;
                break;
            }
			usleep(1000);
            continue;
		}
		
        sort(queueDetOut.front().img , queueDetOut.front().dets.results);  // 会更新 dets.results
        queueOutput.push(queueDetOut.front());
        // showDetection(queueDetOut.front().img, queueDetOut.front().dets.results);
        queueDetOut.pop();
    }
    cout << "Track is over." << endl;
    return 0;
}

void DeepSort::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 2);
        //std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
		//std::string lbl = cv::format("ID:%d_C:%d", (int)box.trackID, (int)box.classID);
		std::string lbl = cv::format("ID:%d",(int)box.trackID);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0,255,0));
    }
    cv::imwrite("./display.jpg", temp);
}