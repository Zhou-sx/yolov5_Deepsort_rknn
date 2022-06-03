#include "common.h"
#include "KalmanTracker.h"
#include "Hungarian.h"

#define CNUM 20

class Sort{
public:
    Sort(int);
    ~Sort();

    void get_color();
    int track_process();

private:
    cv::Scalar_<int> _randColor[CNUM];
    double iouThreshold = 0.3;

    std::vector<KalmanTracker> trackers;
    std::vector<DetectBox> frameTrackingResult;
};

extern double what_time_is_it_now();

double box_iou(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
