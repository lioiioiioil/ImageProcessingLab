#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Гауссов шум
double generateGaussianNoise(double sigma) {
    static bool hasSpare = false;
    static double spare;
    if (hasSpare) {
        hasSpare = false;
        return spare * sigma;
    }
    hasSpare = true;
    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return u * s * sigma;
}

// гистограмы
void saveHistogramImage(const long* hist, const string& name) {
    int h_w = 512, h_h = 400;
    Mat histImg(h_h, h_w, CV_8UC3, Scalar(255, 255, 255)); // Белый фон

    long max_val = 0;
    for (int i = 0; i < 256; i++) if (hist[i] > max_val) max_val = hist[i];

    if (max_val == 0) return;

    for (int i = 0; i < 256; i++) {
        int bin_h = (int)(((double)hist[i] / max_val) * (h_h - 20));
        for (int x = i * 2; x < (i * 2) + 2; x++) {
            for (int y = h_h - 1; y >= h_h - bin_h; y--) {
                histImg.at<Vec3b>(y, x) = Vec3b(150, 50, 50); 
            }
        }
    }
    imwrite("hist_" + name + ".png", histImg);
}

//обработка изображения
void processImage(const string& imagePath, ofstream& logFile) {
    Mat img = imread(imagePath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Ошибка загрузки: " << imagePath << endl;
        return;
    }

    string baseName = imagePath.substr(0, imagePath.find_last_of('.'));
    
    logFile << "========================================" << endl;
    logFile << "Файл: " << imagePath << endl;

    int rows = img.rows;
    int cols = img.cols;
    double total_pixels = (double)rows * cols;

    // 1. Гистограмма
    long hist[256] = {0};
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            hist[img.at<uchar>(r, c)]++;

    saveHistogramImage(hist, baseName);

    // 2. Статистические квартили (добавленный блок)
    int q1 = -1, q2 = -1, q3 = -1;
    long cumulative_sum = 0;
    for (int i = 0; i < 256; i++) {
        cumulative_sum += hist[i];
        if (q1 == -1 && cumulative_sum >= total_pixels * 0.25) q1 = i;
        if (q2 == -1 && cumulative_sum >= total_pixels * 0.50) q2 = i;
        if (q3 == -1 && cumulative_sum >= total_pixels * 0.75) q3 = i;
    }

    // 3. Основные метрики
    double mean = 0, energy = 0, entropy = 0, prob[256];
    for (int i = 0; i < 256; i++) {
        prob[i] = (double)hist[i] / total_pixels;
        mean += i * prob[i];
        energy += prob[i] * prob[i];
        if (prob[i] > 1e-12) entropy -= prob[i] * (log(prob[i]) / log(2.0));
    }

    double variance = 0, m3 = 0, m4 = 0;
    for (int i = 0; i < 256; i++) {
        double diff = i - mean;
        variance += diff * diff * prob[i];
        m3 += pow(diff, 3) * prob[i];
        m4 += pow(diff, 4) * prob[i];
    }
    double std_dev = sqrt(variance);
    double skewness = (std_dev > 0) ? m3 / pow(std_dev, 3) : 0;
    double kurtosis = (std_dev > 0) ? (m4 / pow(std_dev, 4)) - 3.0 : 0;

    logFile << "Ср. яркость: " << mean << "\nДисперсия: " << variance << endl;
    logFile << "Квартиль Q1 (25%): " << q1 << endl;
    logFile << "Квартиль Q2 (медиана): " << q2 << endl;
    logFile << "Квартиль Q3 (75%): " << q3 << endl;
    logFile << "Асимметрия: " << skewness << "\nЭксцесс: " << kurtosis 
            << "\nЭнтропия: " << entropy << "\nЭнергия: " << energy << endl;

    // 4. GLCM (Матрица совместной встречаемости)
    Mat glcm_mat = Mat::zeros(256, 256, CV_64F);
    int dr = 1, dc = 1; 
    double pairs = 0;
    for (int r = 0; r < rows - dr; r++) {
        for (int c = 0; c < cols - dc; c++) {
            glcm_mat.at<double>(img.at<uchar>(r, c), img.at<uchar>(r + dr, c + dc))++;
            pairs++;
        }
    }
    glcm_mat /= pairs;

    // Энергия GLCM
    double glcm_energy = 0;
    for(int i=0; i<256; i++)
        for(int j=0; j<256; j++)
            glcm_energy += pow(glcm_mat.at<double>(i, j), 2);
    
    logFile << "Энергия GLCM: " << glcm_energy << endl;

    double maxGLCM;
    minMaxLoc(glcm_mat, NULL, &maxGLCM);
    Mat glcm_view;
    if(maxGLCM > 0) glcm_mat.convertTo(glcm_view, CV_8U, 255.0 / maxGLCM);
    else glcm_view = Mat::zeros(256, 256, CV_8U);
    imwrite("glcm_" + baseName + ".png", glcm_view);

    // 5. Шум и PSNR
    vector<double> sigmas = {5.0, 10.0, 25.0, 50.0, 75.0, 100.0};
    logFile << "Анализ PSNR для разных уровней шума:" << endl;
    
    for (double s : sigmas) {
        double mse = 0;
        Mat noisy = img.clone();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int original_val = img.at<uchar>(r, c);
                int noise_val = original_val + (int)generateGaussianNoise(s);
                
                if (noise_val < 0) noise_val = 0;
                if (noise_val > 255) noise_val = 255;
                
                noisy.at<uchar>(r, c) = (uchar)noise_val;
                
                double diff = (double)original_val - noise_val;
                mse += diff * diff;
            }
        }
        mse /= total_pixels;
        double psnr = (mse > 0) ? 10.0 * log10((255.0 * 255.0) / mse) : 99.0;
        
        logFile << "  Sigma=" << s << " -> PSNR=" << fixed << setprecision(2) << psnr << " dB" << endl;
        
        string noiseFileName = "noisy_s" + to_string((int)s) + "_" + baseName + ".png";
        imwrite(noiseFileName, noisy);
    }
    logFile << endl;
}

int main() {
    srand(time(0));
    ofstream logFile("lab1_results.txt");
    vector<string> images = {"1.jpg", "2.jpg", "3.jpg", "4.jpg"};
    for (const string& path : images) {
        processImage(path, logFile);
    }
    logFile.close();
    return 0;
}