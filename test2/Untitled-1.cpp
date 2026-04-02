#include <opencv2/opencv.hpp>
#include <iostream>
#include <numpy/numpy.h> 

using namespace cv;
using namespace std;

int main() {
    string img_path = "test2.jpg";  
    Mat src_img = imread(img_path, IMREAD_COLOR);
    if (src_img.empty()) {
        cerr << "错误：无法读取图片，请检查路径是否正确！" << endl;
        return -1;
    }

    cout << "===== 图像基本信息 =====" << endl;
    cout << "图像宽度 (cols): " << src_img.cols << endl;
    cout << "图像高度 (rows): " << src_img.rows << endl;
    cout << "图像通道数: " << src_img.channels() << endl;
    cout << "图像数据类型: " << src_img.type() << " (对应CV_8UC3: 8位无符号3通道)" << endl;
    cout << "图像尺寸: " << src_img.size() << endl;
    cout << "=========================" << endl << endl;

    namedWindow("原图", WINDOW_NORMAL);
    imshow("原图", src_img);

    Mat gray_img;
    cvtColor(src_img, gray_img, COLOR_BGR2GRAY);  
    namedWindow("灰度图", WINDOW_NORMAL);
    imshow("灰度图", gray_img);

    imwrite("gray_test.jpg", gray_img);
    cout << "灰度图已保存为: gray_test.jpg" << endl << endl;

    int x = 100, y = 100;  
    if (src_img.channels() == 3) {
        Vec3b pixel = src_img.at<Vec3b>(y, x);
        cout << "原图 (" << y << "," << x << ") 位置像素值 (BGR): "
             << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << endl;
    }
    uchar gray_pixel = gray_img.at<uchar>(y, x);
    cout << "灰度图 (" << y << "," << x << ") 位置像素值: " << (int)gray_pixel << endl << endl;

    Rect roi(0, 0, 200, 200);  
    Mat crop_img = gray_img(roi);
    imwrite("crop_gray_test.jpg", crop_img);
    cout << "左上角200x200裁剪区域已保存为: crop_gray_test.jpg" << endl;
    namedWindow("裁剪区域", WINDOW_NORMAL);
    imshow("裁剪区域", crop_img);

    waitKey(0);
    destroyAllWindows();

    return 0;
}