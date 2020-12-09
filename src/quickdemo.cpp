#include "quickopencv.h"

// 色彩空间转换
void QuickDemo::colorSpace_Demo(cv::Mat& image){
    cv::Mat gray, hsv;
    cv::cvtColor(image, gray,cv::COLOR_BGR2GRAY);
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);  // H：0～180， S：0～255， V：0-255

    cv::namedWindow("gray image", cv::WINDOW_FREERATIO);
    cv::namedWindow("hsv image", cv::WINDOW_FREERATIO);  
    cv::imshow("gray image", gray);
    cv::imshow("hsv image", hsv);
    return;
}

// Mat创建
void QuickDemo::mat_create_demo(cv::Mat &image){
    cv::Mat m1, m2;
    // 深拷贝
    m1 = image.clone();
    image.copyTo(m2);
    //浅拷贝
    cv::Mat m4 = m1;  //两个指针指向相同的内存
    // 创建空白图像
    cv::Mat m3 = cv::Mat::zeros(cv::Size(8,8), CV_8UC3);  // CV_8UC1前不需要加域名cv::
    m3= 127;  // 如多个通道只在第一个通道加127
    m3 = cv::Scalar(127, 127, 127);  // 在对应通道加相应数值， scalar长度可以比changnel长也可以短
    cout << m3 << endl;
    cout << "width: " << m3.cols << endl;
    cout << "height: " << m3.rows << endl;
    cout << "channel: " << m3.channels() << endl;  // channels 是成员函数，不是属性，需要加()
    return;
}

// Mat 遍历访问像素值
void QuickDemo::pixel_visit_demo(cv::Mat &image){
    int w = image.cols;
    int h = image.rows;
    int dims = image.channels();
    //下标访问  数组访问
    /*
    for(int row=0; row < h; row++){
        for(int col=0; col < w; col++){
            if(dims==1){
                int pv = image.at<uchar>(row,col);
                image.at<uchar>(row, col) = 255-pv;
            }
            else if(dims ==3){
                cv::Vec3b bgr = image.at<cv::Vec3b>(row, col); // Vec3b, Vec3i, Vec3f
                image.at<cv::Vec3b>(row, col)[0] = 255-bgr[0];
                image.at<cv::Vec3b>(row, col)[1] = 255-bgr[1];
                image.at<cv::Vec3b>(row, col)[2] = 255-bgr[2];

            }
            
        }
    }
    */
    // 指针访问
    for(int row=0; row<w; row++){
        uchar* current_row = image.ptr<uchar>(row);  // 返回这一行数组指针
        for(int col=0; col<h; col++){
            if(dims==1){
                int pv = *current_row;
                *current_row++ = 255-pv;
            }
            else if (dims==3){
                *current_row = 255-*current_row;
                current_row++;
                *current_row = 255-*current_row;
                current_row++;
                *current_row = 255-*current_row;
                current_row++;
            }
            
        }
    }
    cv::imshow("convert", image);
    return;

}

// Mat 像素的算术操作
void  QuickDemo::operators_demo(cv::Mat &image){
    cv::Mat dst;
    dst = image + cv::Scalar(50, 50, 50);  // 提高图片亮度， 如向加后值大于255 则值保持255
    dst = image / cv::Scalar(2,2,2);  // 对应通道除以相应数值  
    // dst = image * cv::Scalar(2,2,2);  // 乘法不能直接使用*

    cv::Mat m = cv::Mat::zeros(image.size(), image.type());
    m = cv::Scalar(50,50,50);
    cv::add(image,m, dst);
    cv::subtract(image, m, dst);
    m = cv::Scalar(2,2,2);
    cv::multiply(image, m, dst);
    cv::divide(image, m, dst);
    
    cv::imshow("operators", dst);
}

// Mat 像素的位操作
void QuickDemo::bitwise_demo(cv::Mat &image){
    cv::Mat m1 = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);
    cv::Mat m2 = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);
    cv::rectangle(m1, cv::Rect(100, 100, 80, 80), cv::Scalar(255,255,0), -1, cv::LINE_8);  // 画矩形
    cv::rectangle(m2, cv::Rect(150, 150, 80, 80), cv::Scalar(0,255,255), -1, cv::LINE_8);  // 画矩形
    cv::imshow("m1", m1);
    cv::imshow("m2", m2);
    cv::Mat dst;
    cv::bitwise_and(m1,m2, dst);
    cv::bitwise_or(m1,m2, dst);
    cv::bitwise_not(m1, dst);
    cv::imshow("位操作", dst);
}

// 通道分离与合并
void QuickDemo::channels_demo(cv::Mat &image){
    vector<cv::Mat> mv;
    cv::split(image, mv);  
    cv::imshow("蓝色", mv[0]);
    cv::imshow("绿色", mv[1]);
    cv::imshow("红色", mv[2]);

    cv::Mat dst;
    mv[1] = 0;
    mv[2] = 0;
    cv::merge(mv, dst);
    cv::imshow("合并", dst);

    int from_to[] = {0,2,1,1,2,0}; // 0->2, 1->1, 2->0
    cv::mixChannels(&image, 1, &dst, 1, from_to, 3);
    cv::imshow("mix channel", dst);

}

// 像素值统计:最大值,最小值,均值,方差
void QuickDemo::pixel_statistic_demo(cv::Mat &image){
    double minv, maxv;
    cv::Point minLoc, maxLoc;
    vector<cv::Mat> mv;
    cv::split(image, mv);
    cv::minMaxLoc(mv[0], &minv, &maxv, &minLoc, &maxLoc);  // 只能单通道图像
    cout << "min value: " << minv << "  " << minLoc << endl; 
    cout << "max value: " << maxv << "  " << maxLoc <<endl;

    cv::Mat mean, stddev;
    cv::meanStdDev(image, mean, stddev);  //可以是多通道图像
    cout << "mean: " << mean << endl;
    cout << "stddev: " << stddev << endl;
    return;
}

// 画基础几何形状
void QuickDemo::drawing_demo(cv::Mat &image){
    cv::Rect rect;
    rect.x = 100;
    rect.y = 100;
    rect.width = 250;
    rect.height = 300;
    cv::rectangle(image, rect, cv::Scalar(0,0,255), 2, 8, 0);
    cv::circle(image, cv::Point(350, 400), 15, cv::Scalar(0,255, 0), 2, 8, 0);
    cv::line(image, cv::Point(100, 100), cv::Point(350,400), cv::Scalar(255,0,0),2,8);
    cv::imshow("draw", image);
}

// 随机数 随机颜色
void QuickDemo::random_drawing_demo(cv::Mat &image){
    cv::Mat canvas =cv::Mat::zeros(cv::Size(512, 512), CV_8UC3);
    int w = canvas.cols;
    int h = canvas.rows;
    cv::RNG rng(12345);  // opencv 里的随机数
    while(true){
        int c= cv::waitKey(50);
        if(c==27){  //退出
            break;
        }
        int x1 = rng.uniform(0, w);
        int y1 = rng.uniform(0, h);
        int x2 = rng.uniform(0, w);
        int y2 = rng.uniform(0, h);

        int b = rng.uniform(0, 255);
        int g = rng.uniform(0, 255);
        int r = rng.uniform(0, 255);

        canvas = cv::Scalar(0,0,0);  // 每次清空画布
        cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(b, g, r), 1, cv::LINE_AA, 0);
        cv::imshow("随机绘制演示", canvas);
    }
}

// 绘制填充多边形
void QuickDemo::polyline_drawing_demo(cv::Mat &image){
    cv::Mat canvas = cv::Mat::zeros(cv::Size(512, 512), CV_8UC3);
    cv::Point p1(100, 100);
    cv::Point p2(350, 100);
    cv::Point p3(450, 280);
    cv::Point p4(320, 450);
    cv::Point p5(80, 400);

    vector<cv::Point> pts;
    pts.push_back(p1);
    pts.push_back(p2);
    pts.push_back(p3);
    pts.push_back(p4);
    pts.push_back(p5);

    // cv::polylines(canvas, pts, true, cv::Scalar(0, 0, 255), 2, 8, 0);  // 只能绘制边框， 不能填充 

    vector<vector<cv::Point>> contours;
    contours.push_back(pts);
    cv::drawContours(canvas, contours, -1, cv::Scalar(0,0,255), 1); // thickness 为-1 则填充, 大于等于0则画边线
    cv::imshow("多边形绘制", canvas);

}

// 归一化
void QuickDemo::norm_demo(cv::Mat &image){
    cv::Mat dst;
    image.convertTo(dst, CV_32F);  // 转数据类型
    cout << image.type() << endl;  // 输出16
    cout << dst.type() << endl;  // 输出21
    cv::normalize(dst, dst, 1.0, 0, cv::NORM_MINMAX);
    cv::imshow("归一化图像", dst);  // imshow 可以显示浮点数mat，但取值范围在0～1之间
}

// 图像resize
void QuickDemo::resize_demo(cv::Mat &image){
    cv::Mat zoomin, zoomout;
    int h = image.rows;
    int w = image.cols;
    cv::resize(image, zoomin, cv::Size(w/2, h/2), 0, 0, cv::INTER_LINEAR);
    cv::imshow("zoomin", zoomin);
    cv::resize(image, zoomout, cv::Size(w*1.5, h*1.5), 0, 0, cv::INTER_LINEAR);
    cv::imshow("zoomout", zoomout);
}

// 图像翻转
void QuickDemo::flip_demo(cv::Mat &image){

    cv::Mat dst;
    cv::flip(image, dst, -1);  // 0:上下翻转  1：左右翻转  -1: 上下左右翻转，相当于转180度
    cv::imshow("图像翻转", dst);
}

// 图像旋转
void QuickDemo::rotate_demo(cv::Mat &image){
    cv::Mat dst, M;
    int w = image.cols;
    int h = image.rows;

    M = cv::getRotationMatrix2D(cv::Point2f(w/2, h/2), 45, 1.0);
    // 计算旋转后所需要的长宽
    double cos = abs(M.at<double>(0, 0));
    double sin = abs(M.at<double>(0, 1));
    int nw = cos*w + sin*h;
    int nh = sin*w + cos*h;
    M.at<double>(0,2) = M.at<double>(0, 2) + (nw/2 - w/2);
    M.at<double>(1,2) = M.at<double>(1, 2) + (nh/2 - h/2);
    cv::warpAffine(image, dst, M, cv::Size(nw, nh), cv::INTER_LINEAR, 0, cv::Scalar(255, 0, 0));  // 图像边界填充固定值
    cout << image.channels() << endl;
    cout << image.type() << endl;
    cout << CV_8UC3 << endl;

    cv::imshow("图像旋转", dst);
} 

// 直方图
void QuickDemo::histogram_demo(cv::Mat &image){
    // 三通道分离
    vector<cv::Mat> bgr_plane;
    cv::split(image, bgr_plane);
    // 定义参数变量
    //const int channels[1] = {0};
    const int bins[1] = {256};
    float hranges[2] = {0, 255};
    const float* ranges[1] = {hranges};
    cv::Mat b_hist;
    cv::Mat g_hist;
    cv::Mat r_hist;
    // 计算Blue, Green, Red 通道的直方图
    cv::calcHist(&bgr_plane[0], 1, 0, cv::Mat(), b_hist, 1, bins, ranges);
    cv::calcHist(&bgr_plane[1], 1, 0, cv::Mat(), g_hist, 1, bins, ranges);
    cv::calcHist(&bgr_plane[2], 1, 0, cv::Mat(), r_hist, 1, bins, ranges);
    // 显示直方图
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w/bins[0]);
    cv::Mat histImage = cv::Mat::zeros(hist_h, hist_w, CV_8UC3);
    // 归一化直方图数据
    cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    // 绘制直方图
    for(int i=1; i<bins[0]; i++){
        cv::line(histImage, cv::Point(bin_w*(i-1), hist_h-cvRound(b_hist.at<float>(i-1))),
                cv::Point(bin_w*(i), hist_h-cvRound(b_hist.at<float>(i))), cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w*(i-1), hist_h-cvRound(g_hist.at<float>(i-1))),
                cv::Point(bin_w*(i), hist_h-cvRound(g_hist.at<float>(i))), cv::Scalar(0,255, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w*(i-1), hist_h-cvRound(r_hist.at<float>(i-1))),
                cv::Point(bin_w*(i), hist_h-cvRound(r_hist.at<float>(i))), cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    // 显示直方图
    cv::imshow("直方图", histImage);
}

// 二维直方图
void QuickDemo::histogram_2d_demo(cv::Mat &image){
    cv::Mat hsv, hs_hist;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    int hbins = 30, sbins=32;
    int hist_bins[] = {hbins, sbins};
    float h_range[] = {0, 180};
    float s_range[] = {0, 256};
    const float* hs_ranges[] = {h_range, s_range};
    int hs_channels[] = {0, 1};
    cv::calcHist(&hsv, 1, hs_channels, cv::Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);
    double maxVal = 0;
    cv::minMaxLoc(hs_hist, 0, &maxVal, 0, 0);
    int scale = 10;
    cv::Mat hist2d_image = cv::Mat::zeros(sbins*scale, hbins*scale, CV_8UC3);
    for(int h=0; h<hbins; h++){
        for(int s=0; s<sbins; s++){
            float binVal = hs_hist.at<float>(h,s);
            int intensity = cvRound(binVal*255/maxVal);
            cv::rectangle(hist2d_image, cv::Point(h*scale, s*scale), cv::Point((h+1)*scale-1, (s+1)*scale-1), cv::Scalar::all(intensity), -1); 
        }
    }
    cv::Mat colormap;
    cv::applyColorMap(hist2d_image, colormap, cv::COLORMAP_JET);
    cv::imshow("二维直方图", hist2d_image);
    cv::imshow("二维直方图-彩色", colormap);
}

// 直方图均衡画
void QuickDemo::histgram_eq_demo(cv::Mat &image){
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat dst;
    cv::equalizeHist(gray, dst);  // 只支持单通道，所以需要先转为灰度图
    cv::imshow("未均衡化", gray);
    cv::imshow("直方图均衡化", dst);
}

// 均值模糊(卷积操作)
void QuickDemo::blur_demo(cv::Mat &image){
    cv::Mat dst;
    cv::blur(image, dst, cv::Size(3,3),cv::Point(-1,-1));
    cv::imshow("均值模糊", dst);
}

// 高斯模糊
void QuickDemo::gussian_demo(cv::Mat &image){
    cv::Mat dst;
    // size 卷积核一定是奇数。
    // 如果窗口大小设置合法则会根据窗口大小计算一个sigma， 设置的sigma无效。
    // 如窗口大小设置为0， 则根据sigma计算窗口大小
    cv::GaussianBlur(image, dst, cv::Size(0,0), 3);  
    cv::imshow("高斯模糊", dst);
}

// 高斯双边模糊
void QuickDemo::bifilter_demo(cv::Mat &image){
    cv::Mat dst;
    cv::bilateralFilter(image, dst, 0, 50, 10);
    cv::imshow("双边模糊", dst);
}
