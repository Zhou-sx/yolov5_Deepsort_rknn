#include <unistd.h>
#include <string>
#include <iostream>

#include <thread>

#include "deepsort.h"
#include "common.h"
#include "mytime.h"
using namespace std;

struct video_property;
extern queue<imageout_idx> queueDetOut;  // output queue
extern queue<imageout_idx> queueOutput;  		 // output queue 目标追踪输出队列
extern video_property video_probs;       // 视频属性类
extern bool bDetecting;                  // 目标检测进程状态
extern bool bTracking;                   // 目标追踪进程状态               
extern double end_time;                  // 整个视频追踪结束

extern mutex mtxQueueOutput;
extern mutex mtxQueueDetOut;

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

    // two Re-ID networks, share same CPU and NPU
    featureExtractor1 = new FeatureTensor(enginePath.c_str(), cpu_id, npu_id, 1, 1);
    featureExtractor1->init(imgShape, featureDim, NET_INPUTCHANNEL);

    featureExtractor2 = new FeatureTensor(enginePath.c_str(), cpu_id, npu_id, 1, 1);
    featureExtractor2->init(imgShape, featureDim, NET_INPUTCHANNEL);

}

DeepSort::~DeepSort() {
    delete objTracker;
}

void DeepSort::sort(cv::Mat& frame, vector<DetectBox>& dets) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;  // DETECTIONS: std::vector<DETECTION_ROW> in model.hpp
    vector<CLSCONF> clsConf;
    
    // read every detections in current frame and 
    // store them in detections(bbox) and clsConf(conf scores)
    for (DetectBox i : dets) {  
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.confidence));
    }
    
    result.clear();  // result: vector<pair<int, DETECTBOX>>
    results.clear();  // results: vector<pair<CLSCONF, DETECTBOX>>
    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);  // sort
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
    bool flag = featureExtractor1->getRectsFeature(frame, detections);
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

void DeepSort::sort_interval(cv::Mat& frame, vector<DetectBox>& dets) {
    /*
    If frame_id % this->track_interval != 0, there is no new detections
    so only predict the tracks using Kalman
    */
    if (!dets.empty()) cout << "Error occured! \n";

    result.clear();
    results.clear();
    objTracker->predict();  // Kalman predict

    // update result and results
    // cout << "---------" << objTracker->tracks.size() << "\n";
    for (Track& track : objTracker->tracks) {
            // if (!track.is_confirmed() || track.time_since_update > 1)
            if (!track.is_confirmed() || track.time_since_update > this->track_interval + 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh()));
    }
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

void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;  // std::vector<DETECTION_ROW>

    int numOfDetections = detections.size();
    bool flag1 = true, flag2 = true;
    if (numOfDetections < 2){
        // few objects, use single Re-ID 
        double timeBeforeReID = what_time_is_it_now();
        flag1 = featureExtractor1->getRectsFeature(frame, detections);
        double timeAfterReID = what_time_is_it_now();

        cout << "--------Time cost in ReID: " << timeAfterReID - timeBeforeReID << "\n";
        flag2 = true;
    }
    else {
        DETECTIONS detectionsPart1, detectionsPart2;
        int border = numOfDetections >> 1;
        auto start = detections.begin(), end = detections.end();  // iterator

        double timeBeforeAssign = what_time_is_it_now();
        detectionsPart1.assign(start, start + border);
        detectionsPart2.assign(start + border, end);
        double timeAfterAssign = what_time_is_it_now();

        cout << "--------Time cost in assign: " << timeAfterAssign - timeBeforeAssign << "\n";

        // NOTE: convert pointer or set global variables
        // inference separately
        double timeBeforeReID = what_time_is_it_now();
        thread reID1Thread1 (&FeatureTensor::getRectsFeature, featureExtractor1, std::ref(frame), std::ref(detectionsPart1));
        thread reID1Thread2 (&FeatureTensor::getRectsFeature, featureExtractor2, std::ref(frame), std::ref(detectionsPart2));

        reID1Thread1.join(); reID1Thread2.join();

        double timeAfterReID = what_time_is_it_now();

        cout << "--------Time cost in ReID: " << timeAfterReID - timeBeforeReID << "\n";

        // copy new feature to origin detections

        double timeBeforeUpdateFeatures = what_time_is_it_now();
        for (int idx = 0; flag1 && flag2 && idx < numOfDetections; idx ++) {
            if (idx < border)
                detections[idx].updateFeature(detectionsPart1[idx].feature);
            else 
                detections[idx].updateFeature(detectionsPart2[idx - border].feature);
        }
        double timeAfterUpdateFeatures = what_time_is_it_now();
        cout << "--------Time cost in update features: " << timeAfterUpdateFeatures - timeBeforeUpdateFeatures << "\n";

    }
 
    // bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag1 && flag2) {
        objTracker->predict();
        // std::cout << "In: \n"; 
        objTracker->update(detectionsv2);     
        // std::cout << "Out: \n";    
        result.clear();
        results.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh()));
        }
    }
    else cout << "Re-ID1 Error? " << flag1 << " Re-ID2 Error? " << flag2 << "\n";
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

        // get frame index of queueDetOut.front
        int curFrameIdx = queueDetOut.front().dets.id;
        // cout << "Is id match with result " << (!queueDetOut.front().dets.results.empty() && !(curFrameIdx % 3)) << "\n";
		
        if (curFrameIdx < this->track_interval || !(curFrameIdx % this->track_interval))  // have detections
            sort(queueDetOut.front().img , queueDetOut.front().dets.results);  // 会更新 dets.results
        else  
            sort_interval(queueDetOut.front().img , queueDetOut.front().dets.results);
        mtxQueueOutput.lock();
        // cout << "--------------" << queueDetOut.front().dets.results.size() << "\n";
        queueOutput.push(queueDetOut.front());
        mtxQueueOutput.unlock();
        // showDetection(queueDetOut.front().img, queueDetOut.front().dets.results);
        mtxQueueDetOut.lock();
        queueDetOut.pop();
        mtxQueueDetOut.unlock();
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