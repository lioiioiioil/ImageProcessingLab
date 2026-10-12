#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <filesystem>
#include <iomanip>
#include <cassert>

namespace fs = std::filesystem;

inline int clamp(int v, int min_v, int max_v) {
    return std::max(min_v, std::min(max_v, v));
}

// Безопасное чтение пикселя с репликацией границ (Лекция 3, слайд 23)
inline uint8_t getPixel(const cv::Mat& img, int r, int c) {
    r = clamp(r, 0, img.rows - 1);
    c = clamp(c, 0, img.cols - 1);
    return img.at<uint8_t>(r, c);
}

// Преобразование бинарной маски апертуры в список смещений
std::vector<std::pair<int, int>> getOffsetsFromMask(const cv::Mat& mask) {
    std::vector<std::pair<int, int>> offsets;
    int cy = mask.rows / 2;
    int cx = mask.cols / 2;
    for (int r = 0; r < mask.rows; ++r) {
        for (int c = 0; c < mask.cols; ++c) {
            if (mask.at<uint8_t>(r, c) != 0) {
                offsets.emplace_back(r - cy, c - cx);
            }
        }
    }
    return offsets;
}

// Создание стандартных апертур
cv::Mat createSquareMask(int radius) {
    int size = 2 * radius + 1;
    return cv::Mat::ones(size, size, CV_8U) * 255;
}

cv::Mat createCrossMask(int radius) {
    int size = 2 * radius + 1;
    cv::Mat mask = cv::Mat::zeros(size, size, CV_8U);
    for (int i = 0; i < size; ++i) {
        mask.at<uint8_t>(radius, i) = 255;
        mask.at<uint8_t>(i, radius) = 255;
    }
    return mask;
}

void createDir(const std::string& path) {
    fs::create_directories(path);
}
//ШУМЫ
cv::Mat addAWGN(const cv::Mat& src, double sigma) {
    cv::Mat dst = src.clone();
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, sigma);
    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            double val = dst.at<uint8_t>(r, c) + dist(rng);
            dst.at<uint8_t>(r, c) = cv::saturate_cast<uint8_t>(val);
        }
    }
    return dst;
}

cv::Mat addImpulseNoise(const cv::Mat& src, double prob) {
    cv::Mat dst = src.clone();
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            double p = dist(rng);
            if (p < prob / 2.0) dst.at<uint8_t>(r, c) = 0;
            else if (p < prob)  dst.at<uint8_t>(r, c) = 255;
        }
    }
    return dst;
}


double calcMSE(const cv::Mat& I1, const cv::Mat& I2) {
    assert(I1.size() == I2.size() && I1.type() == I2.type());
    double mse = 0.0;
    for (int r = 0; r < I1.rows; ++r) {
        for (int c = 0; c < I1.cols; ++c) {
            double diff = I1.at<uint8_t>(r, c) - I2.at<uint8_t>(r, c);
            mse += diff * diff;
        }
    }
    return mse / (I1.rows * I1.cols);
}

double calcPSNR(const cv::Mat& I1, const cv::Mat& I2) {
    double mse = calcMSE(I1, I2);
    if (mse < 1e-10) return 100.0; // Практическая бесконечность
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}



// ФИЛЬТРЫ (Лекция 4)
// Ранговая фильтрация
cv::Mat rankFilter(const cv::Mat& src, const std::vector<std::pair<int, int>>& offsets, int k) {
    cv::Mat dst(src.size(), src.type());
    std::vector<uint8_t> buf;
    buf.reserve(offsets.size());

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            buf.clear();
            for (auto [dr, dc] : offsets) {
                buf.push_back(getPixel(src, r + dr, c + dc));
            }
            std::sort(buf.begin(), buf.end());
            int idx = clamp(k, 0, (int)buf.size() - 1);
            dst.at<uint8_t>(r, c) = buf[idx];
        }
    }
    return dst;
}

// Медианный фильтр 
cv::Mat medianFilter(const cv::Mat& src, const std::vector<std::pair<int, int>>& offsets) {
    return rankFilter(src, offsets, offsets.size() / 2);
}

// Усредняющий фильтр 
cv::Mat meanFilter(const cv::Mat& src, const std::vector<std::pair<int, int>>& offsets) {
    cv::Mat dst(src.size(), src.type());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            double sum = 0;
            for (auto [dr, dc] : offsets) {
                sum += getPixel(src, r + dr, c + dc);
            }
            dst.at<uint8_t>(r, c) = cv::saturate_cast<uint8_t>(sum / offsets.size());
        }
    }
    return dst;
}

// Фильтр усечённого среднего 
cv::Mat trimmedMeanFilter(const cv::Mat& src, const std::vector<std::pair<int, int>>& offsets, int d) {
    cv::Mat dst(src.size(), src.type());
    std::vector<uint8_t> buf;
    buf.reserve(offsets.size());

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            buf.clear();
            for (auto [dr, dc] : offsets) {
                buf.push_back(getPixel(src, r + dr, c + dc));
            }
            std::sort(buf.begin(), buf.end());
            int n = buf.size();
            int trim = d / 2;
            if (trim * 2 >= n) trim = (n - 1) / 2; // Защита от переусечения

            double sum = 0;
            int count = 0;
            for (int i = trim; i < n - trim; ++i) {
                sum += buf[i];
                count++;
            }
            dst.at<uint8_t>(r, c) = (count > 0) ? cv::saturate_cast<uint8_t>(sum / count) : buf[n / 2];
        }
    }
    return dst;
}

// МОРФОЛОГИЯ ДЛЯ ПОЛУТОНОВЫХ ИЗОБРАЖЕНИЙ 
cv::Mat erodeGray(const cv::Mat& src, const std::vector<std::pair<int, int>>& offsets) {
    cv::Mat dst(src.size(), src.type());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            uint8_t minVal = 255;
            for (auto [dr, dc] : offsets) {
                minVal = std::min(minVal, getPixel(src, r + dr, c + dc));
            }
            dst.at<uint8_t>(r, c) = minVal;
        }
    }
    return dst;
}

cv::Mat dilateGray(const cv::Mat& src, const std::vector<std::pair<int, int>>& offsets) {
    cv::Mat dst(src.size(), src.type());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            uint8_t maxVal = 0;
            for (auto [dr, dc] : offsets) {
                maxVal = std::max(maxVal, getPixel(src, r + dr, c + dc));
            }
            dst.at<uint8_t>(r, c) = maxVal;
        }
    }
    return dst;
}

cv::Mat openGray(const cv::Mat& src, const std::vector<std::pair<int, int>>& offsets) {
    return dilateGray(erodeGray(src, offsets), offsets);
}

cv::Mat closeGray(const cv::Mat& src, const std::vector<std::pair<int, int>>& offsets) {
    return erodeGray(dilateGray(src, offsets), offsets);
}



int main() {

    const std::vector<std::string> imgNames = {"1.png", "2.png", "3.png", "4.png"};
    const std::string baseDir = "results";
    createDir(baseDir + "/task1_rank");
    createDir(baseDir + "/task2_trimmed");
    createDir(baseDir + "/task3_comparison");
    createDir(baseDir + "/task4_morphology");

    // Апертура: квадрат 3x3 
    cv::Mat maskMat = createSquareMask(1);
    auto offsets = getOffsetsFromMask(maskMat);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "СРАВНЕНИЕ ФИЛЬТРОВ ПО PSNR (дБ)\n";
    std::cout << std::left << std::setw(10) << "Image"
              << std::setw(12) << "Noise"
              << std::setw(12) << "Mean"
              << std::setw(12) << "Median"
              << std::setw(15) << "TrimmedMean" << "\n";
    std::cout << std::string(61, '-') << "\n";

    for (const auto& name : imgNames) {
        cv::Mat src = cv::imread(name, cv::IMREAD_GRAYSCALE);

        // 1. Ранговая фильтрация 
        cv::Mat rankMin = rankFilter(src, offsets, 0);
        cv::Mat rankMax = rankFilter(src, offsets, offsets.size() - 1);
        cv::imwrite(baseDir + "/task1_rank/" + name, rankMin);
        cv::imwrite(baseDir + "/task1_rank/min_" + name, rankMin);
        cv::imwrite(baseDir + "/task1_rank/max_" + name, rankMax);

        // 2. Усечённое среднее
        cv::Mat trimmed = trimmedMeanFilter(src, offsets, 4);
        cv::imwrite(baseDir + "/task2_trimmed/" + name, trimmed);

        // 3. Сравнение PSNR
        cv::Mat noisyAWGN = addAWGN(src, 20.0);
        cv::Mat noisyImpulse = addImpulseNoise(src, 0.05);

        auto processAndCompare = [&](const cv::Mat& noisy, const std::string& noiseType) {
            cv::Mat fMean = meanFilter(noisy, offsets);
            cv::Mat fMedian = medianFilter(noisy, offsets);
            cv::Mat fTrimmed = trimmedMeanFilter(noisy, offsets, 4);

            cv::imwrite(baseDir + "/task3_comparison/" + noiseType + "_" + name, noisy);
            cv::imwrite(baseDir + "/task3_comparison/mean_" + noiseType + "_" + name, fMean);
            cv::imwrite(baseDir + "/task3_comparison/median_" + noiseType + "_" + name, fMedian);
            cv::imwrite(baseDir + "/task3_comparison/trimmed_" + noiseType + "_" + name, fTrimmed);

            std::cout << std::left << std::setw(10) << name
                      << std::setw(12) << noiseType
                      << std::setw(12) << calcPSNR(src, fMean)
                      << std::setw(12) << calcPSNR(src, fMedian)
                      << std::setw(15) << calcPSNR(src, fTrimmed) << "\n";
        };

        processAndCompare(noisyAWGN, "AWGN");
        processAndCompare(noisyImpulse, "Impulse");

        // 4. Морфологические операции
        cv::Mat eroded = erodeGray(src, offsets);
        cv::Mat dilated = dilateGray(src, offsets);
        cv::Mat opened = openGray(src, offsets);
        cv::Mat closed = closeGray(src, offsets);

        cv::imwrite(baseDir + "/task4_morphology/erode_" + name, eroded);
        cv::imwrite(baseDir + "/task4_morphology/dilate_" + name, dilated);
        cv::imwrite(baseDir + "/task4_morphology/open_" + name, opened);
        cv::imwrite(baseDir + "/task4_morphology/close_" + name, closed);
    }
    return 0;
}