#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

/*
Реализовать вычисление прямого быстрого двумерного преобразования Фурье
для одномерных сигналов и двумерных изображений произвольного размера.
Сохранить изображения логарифма амплитудного спектра для сигнала и
изображения в файле графического формата. Низкие частоты должны быть
расположены в центре спектра. Показать примеры работы программы для
смеси сигнала и изображения из нескольких синусоид.
*/

typedef complex<double> cd;
const double PI = acos(-1.0);

void ComplexBitReverse(vector<cd>& data) {
    int n = data.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            swap(data[i], data[j]);
    }
}

void FftDit(vector<cd>& data, int dir = 1) {
    int n = data.size();
    ComplexBitReverse(data);

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * dir;
        cd wlen(cos(ang), -sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = data[i + j], v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

int GetNextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

void test1D() {
    cout << "Running 1D Test..." << endl;
    const int N = 600; 
    int paddedN = GetNextPowerOfTwo(N);
    
    vector<cd> signal(paddedN, cd(0, 0));

    // 10 Гц, 40 Гц и 100 Гц
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        double val = sin(2 * PI * 10 * t) + 0.5 * sin(2 * PI * 40 * t) + 0.2 * sin(2 * PI * 100 * t);
        
        // умножение на (-1)^i
        double shift = (i % 2 == 0) ? 1.0 : -1.0;
        signal[i] = cd(val * shift, 0);
    }

    FftDit(signal);

    vector<double> magnitude(paddedN);
    double maxVal = 0;
    for (int i = 0; i < paddedN; i++) {
        magnitude[i] = log(1.0 + abs(signal[i]));
        if (magnitude[i] > maxVal) maxVal = magnitude[i];
    }

    // Отрисовка спектрограммы
    int graphH = 300;
    Mat spectrumImg(graphH, paddedN, CV_8U, Scalar(255));
    for (int i = 0; i < paddedN; i++) {
        int barHeight = (int)((magnitude[i] / maxVal) * (graphH - 20));
        line(spectrumImg, Point(i, graphH - 1), Point(i, graphH - 1 - barHeight), Scalar(0), 1);
    }

    imwrite("test1d_spectrum.png", spectrumImg);
    cout << "1D Spectrum saved to test1d_spectrum.png" << endl;
}

//Вычисление 2D БПФ 
void Fft2D(vector<vector<cd>>& data) {
    int rows = data.size();
    int cols = data[0].size();

    // БПФ для каждой строки
    for (int i = 0; i < rows; i++) {
        FftDit(data[i]);
    }

    // БПФ для каждого столбца
    for (int j = 0; j < cols; j++) {
        vector<cd> column(rows);
        for (int i = 0; i < rows; i++) column[i] = data[i][j];
        FftDit(column);
        for (int i = 0; i < rows; i++) data[i][j] = column[i];
    }
}

void process2D(const Mat& input, string suffix) {
    int h = input.rows;
    int w = input.cols;
    int ph = GetNextPowerOfTwo(h);
    int pw = GetNextPowerOfTwo(w);

    vector<vector<cd>> data(ph, vector<cd>(pw, cd(0, 0)));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double val = input.at<uchar>(y, x);
            //умножение на (-1)^(x+y)
            double shift = ((x + y) % 2 == 0) ? 1.0 : -1.0;
            data[y][x] = cd(val * shift, 0);
        }
    }

    Fft2D(data);

    // Расчет логарифмической амплитуды
    Mat spectrum(ph, pw, CV_64F);
    double minM, maxM;
    for (int y = 0; y < ph; y++) {
        for (int x = 0; x < pw; x++) {
            double mag = log(1.0 + abs(data[y][x]));
            spectrum.at<double>(y, x) = mag;
        }
    }
    
    minMaxLoc(spectrum, &minM, &maxM);
    Mat result8U;
    spectrum.convertTo(result8U, CV_8U, 255.0 / (maxM - minM), -minM * 255.0 / (maxM - minM));

    imwrite("spectrum_" + suffix + ".png", result8U);
    cout << "2D Spectrum saved to spectrum_" << suffix << ".png" << endl;
}

int main() {
    // 1. Тест на 1D сигнале
    test1D();

    // 2. Тест на синтетическом 2D сигнале (синусоиды)
    Mat synth(300, 450, CV_8U);
    for (int y = 0; y < synth.rows; y++) {
        for (int x = 0; x < synth.cols; x++) {
            double val = 127 + 127 * sin(2 * PI * 5 * x / synth.cols) * sin(2 * PI * 10 * y / synth.rows);
            synth.at<uchar>(y, x) = (uchar)val;
        }
    }
    imwrite("test2d_synth_input.png", synth);
    process2D(synth, "synthetic");

    // 3. Тест на реальных изображениях
    vector<string> images = {"1.jpg", "2.jpg", "3.jpg", "4.jpg", "input.png"};
    for (const string& path : images) {
        Mat realImg = imread(path, IMREAD_GRAYSCALE);
        process2D(realImg, "path_" + path);
    }

    return 0;
}