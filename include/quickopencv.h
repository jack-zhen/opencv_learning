#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;


class QuickDemo{
    public:
    void colorSpace_Demo(cv::Mat& image);  // 色彩空间转换RGB，BGR，HSV
    void mat_create_demo(cv::Mat &image);  // Mat的创建
    void pixel_visit_demo(cv::Mat &image); // Mat 遍历访问像素值
    void operators_demo(cv::Mat &image);  // Mat 像素的算术操作
    void bitwise_demo(cv::Mat &image);  // Mat 像素的位操作
    void channels_demo(cv::Mat &image); // 通道分离与合并
    void pixel_statistic_demo(cv::Mat &image); // 像素值统计:最大值,最小值,均值,方差
    void drawing_demo(cv::Mat &image);  // 画基础几何形状
    void random_drawing_demo(cv::Mat &image); // 随机数 随机颜色
    void polyline_drawing_demo(cv::Mat &image); // 绘制填充多边形
    void norm_demo(cv::Mat &image);  // 归一化
    void resize_demo(cv::Mat &image);  // 图像resize
    void flip_demo(cv::Mat &image);  // 图像翻转
    void rotate_demo(cv::Mat &image);  // 图像旋转
    void histogram_demo(cv::Mat &image);  // 直方图
    void histogram_2d_demo(cv::Mat &image); // 二维直方图
    void histgram_eq_demo(cv::Mat &image); // 直方图均衡画
    void blur_demo(cv::Mat &image);  // 均值模糊(卷积操作)
    void gussian_demo(cv::Mat &image);  // 高斯模糊
    void bifilter_demo(cv::Mat &image); // 高斯双边模糊
};