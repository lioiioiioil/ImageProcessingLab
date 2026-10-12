#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem> 

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

const double PI = 3.14159265358979323846;


//Метод ближайшего соседа 
uchar getNearestNeighbor(const Mat& src, double x, double y) {
    int ix = (int)std::round(x);
    int iy = (int)std::round(y);
    if (ix < 0 || ix >= src.cols || iy < 0 || iy >= src.rows) return 0;
    return src.at<uchar>(iy, ix);
}

//Билинейная интерполяция 
uchar getBilinear(const Mat& src, double x, double y) {
    int x1 = (int)std::floor(x);
    int y1 = (int)std::floor(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    if (x1 < 0 || x2 >= src.cols || y1 < 0 || y2 >= src.rows) return 0;

    double dx = x - x1;
    double dy = y - y1;

    double res = src.at<uchar>(y1, x1) * (1.0 - dx) * (1.0 - dy) +
                 src.at<uchar>(y1, x2) * dx * (1.0 - dy) +
                 src.at<uchar>(y2, x1) * (1.0 - dx) * dy +
                 src.at<uchar>(y2, x2) * dx * dy;
    return (uchar)res;
}

//Бикубическая интерполяция
double cubicWeight(double s, double a = -0.5) {
    s = std::abs(s);
    if (s <= 1.0) return (a + 2.0) * std::pow(s, 3) - (a + 3.0) * std::pow(s, 2) + 1.0;
    else if (s < 2.0) return a * std::pow(s, 3) - 5.0 * a * std::pow(s, 2) + 8.0 * a * s - 4.0 * a;
    return 0;
}

uchar getBicubic(const Mat& src, double x, double y) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);

    if (x0 < 1 || x0 >= src.cols - 2 || y0 < 1 || y0 >= src.rows - 2) return 0;

    double sum = 0;
    for (int j = -1; j <= 2; j++) {
        double wy = cubicWeight(y - (y0 + j));
        for (int i = -1; i <= 2; i++) {
            double wx = cubicWeight(x - (x0 + i));
            sum += src.at<uchar>(y0 + j, x0 + i) * wx * wy;
        }
    }
    return (uchar)std::max(0.0, std::min(255.0, sum));
}

Mat rotateImageFull(const Mat& src, double angle_deg, string method, double& time_ms) {
    double angle_rad = angle_deg * PI / 180.0;
    double cos_a = std::abs(std::cos(angle_rad));
    double sin_a = std::abs(std::sin(angle_rad));

    int new_w = (int)(src.cols * cos_a + src.rows * sin_a);
    int new_h = (int)(src.cols * sin_a + src.rows * cos_a);

    Mat dst = Mat::zeros(new_h, new_w, CV_8UC1);

    double src_cx = src.cols / 2.0;
    double src_cy = src.rows / 2.0;
    double dst_cx = new_w / 2.0;
    double dst_cy = new_h / 2.0;

    double inv_angle = -angle_rad;
    double inv_cos = std::cos(inv_angle);
    double inv_sin = std::sin(inv_angle);

    auto start = chrono::high_resolution_clock::now();

    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            double xt = x - dst_cx;
            double yt = y - dst_cy;
            double xs = xt * inv_cos - yt * inv_sin + src_cx;
            double ys = xt * inv_sin + yt * inv_cos + src_cy;

            if (method == "nearest")
                dst.at<uchar>(y, x) = getNearestNeighbor(src, xs, ys);
            else if (method == "bilinear")
                dst.at<uchar>(y, x) = getBilinear(src, xs, ys);
            else if (method == "bicubic")
                dst.at<uchar>(y, x) = getBicubic(src, xs, ys);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    time_ms = chrono::duration<double, milli>(end - start).count();

    return dst;
}

int main() {
    vector<string> methods = {"nearest", "bilinear", "bicubic"};
    vector<string> imagePaths = {"1.png", "2.png", "3.png", "4.png"};
    vector<double> angles = {0.0,35.0, 67.0, 100.0, 150.5}; 

    ofstream logFile("TimeStamps.txt");
    logFile << "Method\tImage\tAngle\tTime(ms)\tResolution" << endl;

    for (const string& method : methods) {
        fs::create_directory(method);

        for (const string& imgPath : imagePaths) {
            Mat src = imread(imgPath, IMREAD_GRAYSCALE);

            string imgName = fs::path(imgPath).stem().string(); 
            string subDirPath = method + "/" + imgName;
            fs::create_directories(subDirPath); 

            for (double angle : angles) {
                double time_ms;
                Mat result = rotateImageFull(src, angle, method, time_ms);

                string savePath = subDirPath + "/rotated_" + to_string((int)angle) + ".png";
                imwrite(savePath, result);

                logFile << method << "\t" << imgName << "\t" << angle << "\t" << time_ms << "\t" << result.cols << "x" << result.rows << endl;
            }
        }
        logFile << endl; 
    }

    logFile.close();
    return 0;
}