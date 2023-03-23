#include "rknn_fp.h"


class Yolo :public rknn_fp{
public:
    using rknn_fp::rknn_fp;  //声明使用基类的构造函数
    int detect_process();
private:
    const int det_interval = 1;
};

