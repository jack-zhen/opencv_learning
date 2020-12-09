#include <iostream>
#include "quickopencv.h"

using namespace std;
int main(){
    cv::Mat image = cv::imread("lena.jpg");  // flag默认1 RGB读取， 0 灰度
    if(image.empty()){
        cout << "read image faile" << endl;
        return -1;
    }

    // 显示图片
    cv::namedWindow("input image", cv::WINDOW_FREERATIO);  // 通过窗口名设定窗口性质，可以拖动窗体大小
    cv::imshow("input image", image);

    QuickDemo demo;
    // demo.colorSpace_Demo(image);  // 色彩空间转换
    // demo.mat_create_demo(image);  // 创建Mat
    // demo.pixel_visit_demo(image);  // Mat 遍历访问像素值
    // demo.operators_demo(image);  //Mat 像素的算术操作
    // demo.bitwise_demo(image);  // Mat 像素的位操作
    // demo.channels_demo(image);  // 通道分离与合并
    // demo.pixel_statistic_demo(image);  // 像素值统计:最大值,最小值,均值,方差
    // demo.drawing_demo(image); //画基础几何形状
    // demo.random_drawing_demo(image);  // 随机数
    // demo.polyline_drawing_demo(image); // 绘制多边形
    // demo.norm_demo(image);  // 归一化
    // demo.resize_demo(image); // 图像缩放 
    // demo.flip_demo(image); // 图像翻转
    // demo.rotate_demo(image);  // 图像旋转
    // demo.histogram_demo(image);  // 直方图
    // demo.histogram_2d_demo(image); // 二维直方图
    // demo.histgram_eq_demo(image);  // 直方图均衡画
    // demo.blur_demo(image);  // 均值模糊
    // demo.gussian_demo(image);  // 高斯模糊
    demo.bifilter_demo(image); // 高斯双边模糊
    cv::waitKey(0);  // 0是一致阻塞，其他数值表示阻塞多少毫秒
    return 0;
}