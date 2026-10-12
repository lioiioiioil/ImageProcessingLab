#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <ctime>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

uchar manual_clamp(double val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return static_cast<uchar>(val);
}

// ГЕНЕРАЦИЯ ГАУССОВА ШУМА
double generateGaussianNoise(double mean, double sigma) {
    static double z1;
    static bool generate = false;
    generate = !generate;
    if (!generate) return z1 * sigma + mean;

    double u1, u2;
    do {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= 1e-7);

    double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    z1 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);
    return z0 * sigma + mean;
}

// УНИВЕРСАЛЬНАЯ ДВУМЕРНАЯ СВЕРТКА 
cv::Mat convolveRaw(const cv::Mat& src, const std::vector<std::vector<double>>& kernel) {
    int kSize = (int)kernel.size();
    int kHalf = kSize / 2;
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_64FC1);

    for (int y = kHalf; y < src.rows - kHalf; ++y) {
        for (int x = kHalf; x < src.cols - kHalf; ++x) {
            double sum = 0.0;
            for (int i = 0; i < kSize; ++i) {
                for (int j = 0; j < kSize; ++j) {
                    sum += src.at<uchar>(y - kHalf + i, x - kHalf + j) * kernel[i][j];
                }
            }
            dst.at<double>(y, x) = sum;
        }
    }
    return dst;
}

// УСРЕДНЯЮЩИЙ ФИЛЬТР С ПОРОГОМ
cv::Mat thresholdAveraging(const cv::Mat& src, int kSize, double T) {
    cv::Mat dst = src.clone();
    int kHalf = kSize / 2;
    double n = (double)kSize * kSize;

    for (int y = kHalf; y < src.rows - kHalf; ++y) {
        for (int x = kHalf; x < src.cols - kHalf; ++x) {
            double sum = 0.0;
            for (int i = -kHalf; i <= kHalf; ++i) {
                for (int j = -kHalf; j <= kHalf; ++j) {
                    sum += src.at<uchar>(y + i, x + j);
                }
            }
            double avg = sum / n;
            if (std::abs(src.at<uchar>(y, x) - avg) > T) {
                dst.at<uchar>(y, x) = manual_clamp(avg);
            }
        }
    }
    return dst;
}

// ПОИСК ПЕРЕСЕЧЕНИЯ НУЛЕВОГО УРОВНЯ — УЛУЧШЕННАЯ ВЕРСИЯ
cv::Mat zeroCrossing(const cv::Mat& signedImage, double T) {
    cv::Mat dst = cv::Mat::zeros(signedImage.size(), CV_8UC1);
    for (int y = 1; y < signedImage.rows - 1; ++y) {
        for (int x = 1; x < signedImage.cols - 1; ++x) {
            double val = signedImage.at<double>(y, x);
            double neighbors[4] = {
                signedImage.at<double>(y - 1, x), 
                signedImage.at<double>(y + 1, x),
                signedImage.at<double>(y, x - 1), 
                signedImage.at<double>(y, x + 1)
            };
            
            for (double n : neighbors) {
                // Проверяем пересечение нуля (разные знаки)
                if (val * n < 0) {
                    // Сила градиента — разница между значениями
                    double gradient = std::abs(val - n);
                    // Контраст — максимальное отклонение от нуля
                    double magnitude = std::max(std::abs(val), std::abs(n));
                    
                    // Только сильные, явные границы: высокий градиент И высокий контраст
                    if (gradient > T * 2.0 && magnitude > T * 3.0) {
                        dst.at<uchar>(y, x) = 255;
                        break;
                    }
                }
            }
        }
    }
    return dst;
}

// РУЧНОЕ ДОБАВЛЕНИЕ ШУМА
void applyNoise(const cv::Mat& src, cv::Mat& outGauss, cv::Mat& outSalt) {
    outGauss = src.clone();
    outSalt = src.clone();
    // Гауссов
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            outGauss.at<uchar>(y, x) = manual_clamp(src.at<uchar>(y, x) + generateGaussianNoise(0, 25));
        }
    }
    // Импульсный (5% пикселей)
    for (int i = 0; i < src.rows * src.cols * 0.05; ++i) {
        outSalt.at<uchar>(rand() % src.rows, rand() % src.cols) = (rand() % 2 == 0) ? 255 : 0;
    }
}

// Преобразование матрицы 
cv::Mat to8U(const cv::Mat& src, bool shift = false) {
    cv::Mat dst(src.size(), CV_8UC1);
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            double v = src.at<double>(r, c);
            dst.at<uchar>(r, c) = manual_clamp(shift ? v + 128 : v);
        }
    }
    return dst;
}

// Вспомогательная функция для замера времени
double getElapsedMs(const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    std::vector<std::string> images = {"1.png", "2.png", "3.png", "4.png"};
    std::vector<std::string> subfolders = {
        "1_Noise", "2_LPF_Gaussian", "3_Threshold_Averaging", 
        "4_HPF_LoG", "5_Sharpening", "6_Edge_Detection"
    };

    for (const auto& folder : subfolders) {
        fs::create_directories("results/" + folder);
    }

    // Открытие файла для записи результатов
    std::ofstream conclusion("Conclusion.txt");
    conclusion << "=== ВРЕМЯ ВЫПОЛНЕНИЯ ФИЛЬТРОВ (мс) ===" << std::endl << std::endl;

    // Определения ядер 
    std::vector<std::vector<double>> boxKernel = {{1/9., 1/9., 1/9.}, {1/9., 1/9., 1/9.}, {1/9., 1/9., 1/9.}};
    std::vector<std::vector<double>> laplaceKernel = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
    std::vector<std::vector<double>> logKernel = {
        {0,0,-1,0,0}, {0,-1,-2,-1,0}, {-1,-2,16,-2,-1}, {0,-1,-2,-1,0}, {0,0,-1,0,0}
    };
    std::vector<std::vector<double>> sharpKernel = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};

    for (const std::string& fileName : images) {
        cv::Mat src = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
        if (src.empty()) {
            std::cout << "File " << fileName << " not found, skipping..." << std::endl;
            continue;
        }

        conclusion << "Изображение: " << fileName << std::endl;
        auto totalStart = std::chrono::high_resolution_clock::now();

        //  1. ШУМ 
        auto t1 = std::chrono::high_resolution_clock::now();
        cv::Mat noisyGauss, noisySalt;
        applyNoise(src, noisyGauss, noisySalt);
        double t_noise = getElapsedMs(t1);
        cv::imwrite("results/1_Noise/gauss_" + fileName, noisyGauss);
        cv::imwrite("results/1_Noise/salt_" + fileName, noisySalt);
        conclusion << "  Noise generation: " << t_noise << " ms" << std::endl;

        //  2. ФНЧ
        auto t2 = std::chrono::high_resolution_clock::now();
        cv::Mat lpfBox = convolveRaw(noisyGauss, boxKernel);
        double t_lpf = getElapsedMs(t2);
        cv::imwrite("results/2_LPF_Gaussian/box_filtered_" + fileName, to8U(lpfBox));
        conclusion << "  LPF (Box 3x3): " << t_lpf << " ms" << std::endl;

        //  3. Усредняющий с порогом 
        auto t3 = std::chrono::high_resolution_clock::now();
        cv::Mat nonlinear = thresholdAveraging(noisySalt, 3, 35.0);
        double t_threshold = getElapsedMs(t3);
        cv::imwrite("results/3_Threshold_Averaging/threshold_filtered_" + fileName, nonlinear);
        conclusion << "  Threshold Averaging: " << t_threshold << " ms" << std::endl;

        //  4. ФВЧ и LoG 
        auto t4 = std::chrono::high_resolution_clock::now();
        cv::Mat rawLaplace = convolveRaw(src, laplaceKernel);
        double t_laplace = getElapsedMs(t4);
        
        auto t5 = std::chrono::high_resolution_clock::now();
        cv::Mat rawLoG = convolveRaw(src, logKernel);
        double t_log = getElapsedMs(t5);
        
        cv::imwrite("results/4_HPF_LoG/laplacian_" + fileName, to8U(rawLaplace, true));
        cv::imwrite("results/4_HPF_LoG/log_filter_" + fileName, to8U(rawLoG, true));
        conclusion << "  Laplacian: " << t_laplace << " ms" << std::endl;
        conclusion << "  LoG: " << t_log << " ms" << std::endl;

        //  5. Повышение резкости 
        auto t6 = std::chrono::high_resolution_clock::now();
        cv::Mat sharpened = convolveRaw(src, sharpKernel);
        double t_sharp = getElapsedMs(t6);
        cv::imwrite("results/5_Sharpening/sharp_" + fileName, to8U(sharpened));
        conclusion << "  Sharpening: " << t_sharp << " ms" << std::endl;

        //  6. Детектирование границ 
        auto t7 = std::chrono::high_resolution_clock::now();
        cv::Mat borders = zeroCrossing(rawLoG, 1.5);
        double t_edges = getElapsedMs(t7);
        cv::imwrite("results/6_Edge_Detection/edges_" + fileName, borders);
        conclusion << "  Zero-Crossing Edges: " << t_edges << " ms" << std::endl;

        auto totalEnd = std::chrono::high_resolution_clock::now();
        double total = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
        conclusion << "  >>> TOTAL: " << total << " ms" << std::endl << std::endl;

        std::cout << "Done processing: " << fileName << std::endl;
    }

    conclusion.close();
    std::cout << "\nTiming results saved to Conclusion.txt" << std::endl;
    return 0;
}