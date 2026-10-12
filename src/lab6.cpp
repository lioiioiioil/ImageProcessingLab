#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

struct DSU {
    vector<int> parent;
    DSU(int n) : parent(n) { iota(parent.begin(), parent.end(), 0); }
    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    void unite(int i, int j) {
        int root_i = find(i), root_j = find(j);
        if (root_i != root_j) parent[root_i] = root_j;
    }
};

struct RegionInfo {
    int id;
    double m00;          // Площадь (момент 0-го порядка)
    double m10, m01;     // Моменты 1-го порядка
    double mu20, mu02;   // Центральные моменты 2-го порядка
    double xc, yc;       // Центр масс
    double kr2;          // Коэффициент округлости
    bool isCircle;       // Похож на круг
};

//Алгоритм Оцу
int getOtsuThreshold(const Mat& img) {
    const int HIST_SIZE = 256;
    vector<double> hist(HIST_SIZE, 0.0);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            hist[img.at<uchar>(y, x)]++;

    double total = img.rows * img.cols;
    double sum_all = 0.0;
    for (int i = 0; i < HIST_SIZE; ++i) sum_all += i * hist[i];

    double sum_b = 0.0, w_b = 0.0, w_f = 0.0;
    double max_var = 0.0;
    int threshold = 0;

    for (int t = 0; t < HIST_SIZE; ++t) {
        w_b += hist[t];
        if (w_b == 0) continue;
        w_f = total - w_b;
        if (w_f == 0) break;

        sum_b += t * hist[t];
        double m_b = sum_b / w_b;
        double m_f = (sum_all - sum_b) / w_f;
        double var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f);

        if (var_between > max_var) {
            max_var = var_between;
            threshold = t;
        }
    }
    return threshold;
}

// Превращение в бинарное изображение
Mat binarizeImage(const Mat& img, int th) {
    Mat bin = Mat::zeros(img.size(), CV_8UC1);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            bin.at<uchar>(y, x) = (img.at<uchar>(y, x) > th) ? 255 : 0;
    return bin;
}

// Разметка 4-связных компонент 
//двухпрохходный алгоритм
int label4Connected(const Mat& bin, Mat& labels) {
    labels.create(bin.size(), CV_32SC1);
    labels.setTo(0);
    
    int next_label = 1;
    int max_labels = (bin.rows * bin.cols) / 2 + 2; // Безопасный верхний предел
    DSU dsu(max_labels);

    // ПЕРВЫЙ ПРОХОД
    for (int y = 0; y < bin.rows; ++y) {
        for (int x = 0; x < bin.cols; ++x) {
            if (bin.at<uchar>(y, x) == 0) continue;

            int top  = (y > 0) ? labels.at<int>(y - 1, x) : 0;
            int left = (x > 0) ? labels.at<int>(y, x - 1) : 0;

            if (top == 0 && left == 0) {
                labels.at<int>(y, x) = next_label;
                next_label++;
            } else if (top > 0 && left == 0) {
                labels.at<int>(y, x) = dsu.find(top);
            } else if (top == 0 && left > 0) {
                labels.at<int>(y, x) = dsu.find(left);
            } else {
                int r_top  = dsu.find(top);
                int r_left = dsu.find(left);
                labels.at<int>(y, x) = r_top;
                dsu.unite(r_top, r_left);
            }
        }
    }

    // ВТОРОЙ ПРОХОД
    vector<int> new_label_map(next_label, 0);
    int final_count = 1;
    for (int i = 1; i < next_label; ++i) {
        int root = dsu.find(i);
        if (new_label_map[root] == 0) new_label_map[root] = final_count++;
    }
    final_count--;

    for (int y = 0; y < labels.rows; ++y)
        for (int x = 0; x < labels.cols; ++x)
            if (labels.at<int>(y, x) > 0)
                labels.at<int>(y, x) = new_label_map[dsu.find(labels.at<int>(y, x))];
                
    return final_count;
}

//Вычисление моментов
vector<RegionInfo> computeMomentsAndShape(const Mat& labels, int num_labels) {
    vector<RegionInfo> regions(num_labels + 1);
    vector<vector<pair<int, int>>> pixels(num_labels + 1);
    
    for (int y = 0; y < labels.rows; ++y) {
        const int* row_ptr = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            int lbl = row_ptr[x];
            if (lbl > 0) pixels[lbl].emplace_back(x, y);
        }
    }

    for (int l = 1; l <= num_labels; ++l) {
        const auto& pts = pixels[l];
        if (pts.empty()) continue;

        double m00 = static_cast<double>(pts.size());
        double m10 = 0, m01 = 0;
        for (auto& p : pts) { m10 += p.first; m01 += p.second; }

        double xc = m10 / m00;
        double yc = m01 / m00;

        double mu20 = 0, mu02 = 0;
        for (auto& p : pts) {
            double dx = p.first - xc;
            double dy = p.second - yc;
            mu20 += dx * dx;
            mu02 += dy * dy;
        }

        // Kr2 = 2*pi*(mu20 + mu02) / S^2
        double kr2 = 2.0 * M_PI * (mu20 + mu02) / (m00 * m00);
        bool isCircle = (m00 > 30.0) && (kr2 < 1.05 && kr2 > 0.9);

        regions[l] = {l, m00, m10, m01, mu20, mu02, xc, yc, kr2, isCircle};
    }
    return regions;
}

// Визуализация
void saveCircleHighlight(const Mat& bin, const Mat& labels, const vector<RegionInfo>& regions, 
                         int num_labels, const string& filename) {
    // Создаём 3-канальное изображение на базе бинарного (без cvtColor)
    Mat res(bin.rows, bin.cols, CV_8UC3);
    for (int y = 0; y < bin.rows; ++y) {
        for (int x = 0; x < bin.cols; ++x) {
            uchar v = bin.at<uchar>(y, x);
            res.at<Vec3b>(y, x) = Vec3b(v, v, v); // B=G=R для оттенков серого
        }
    }

    // Закрашиваем круглые области зелёным 
    for (int y = 0; y < labels.rows; ++y) {
        const int* row_lbl = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            int lbl = row_lbl[x];
            if (lbl > 0 && lbl <= num_labels && regions[lbl].isCircle) {
                res.at<Vec3b>(y, x) = Vec3b(0, 255, 0); // Green
            }
        }
    }
    imwrite(filename, res);
}

//База, основа, глава
void processImage(const string& filename) {
    cout << "Processing: " << filename << "..." << endl;
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    if (img.empty()) { cerr << "Ошибка открытия " << filename << endl; return; }

    fs::path p(filename);
    string folder = "out_" + p.stem().string();
    fs::create_directories(folder);

    imwrite(folder + "/" + p.stem().string() + "_grayscale.png", img);

    int th = getOtsuThreshold(img);
    Mat bin = binarizeImage(img, th);
    imwrite(folder + "/" + p.stem().string() + "_binary.png", bin);

    string txt_path = folder + "/" + p.stem().string() + "_report.txt";
    ofstream out(txt_path);
    out << fixed << setprecision(2);
    out << "Image: " << filename << "\nSize: " << img.cols << "x" << img.rows << "\nOtsu Threshold: " << th << "\n\n";
    out << "========================================\n";
    out << "Белые области\n";
    out << "========================================\n";
    Mat labels_white;
    int num_white = label4Connected(bin, labels_white);
    auto info_white = computeMomentsAndShape(labels_white, num_white);
    saveCircleHighlight(bin, labels_white, info_white, num_white, folder + "/" + p.stem().string() + "_white_circles.png");
    
    int cnt_white = 0;
    out << "Всего: " << num_white << "\n";
    out << left << setw(4) << "ID" << setw(10) << "Area" << setw(15) << "Center(X,Y)" << setw(10) << "Kr2" << setw(6) << "Circle\n";
    out << "----------------------------------------------------------------\n";
    for (int l = 1; l <= num_white; ++l) {
        const auto& r = info_white[l];
        if (r.m00 == 0) continue;
        if (r.isCircle) cnt_white++;
        string center_str = "(" + to_string((int)r.xc) + ", " + to_string((int)r.yc) + ")";
    
        out  << r.id 
        << "    " << fixed << setprecision(2) << r.m00 
        << "    " << left << center_str
        << "    " << setw(2) << right << fixed << setprecision(4) << r.kr2 
        << "    " << setw(6) << (r.isCircle ? "YES" : "NO") << "\n";
    }
    out << "Circle-like (Area>30): " << cnt_white << "\n\n";

    //ЧЁРНЫЕ
    out << "========================================\n";
    out << "Чёрные области\n";
    out << "========================================\n";
    Mat bin_inv = ~bin;
    Mat labels_black;
    int num_black = label4Connected(bin_inv, labels_black);
    auto info_black = computeMomentsAndShape(labels_black, num_black);
    saveCircleHighlight(bin, labels_black, info_black, num_black, folder + "/" + p.stem().string() + "_black_circles.png");

    int cnt_black = 0;
    out << "Всего: " << num_black << "\n";
    out << left << setw(4) << "ID" << setw(10) << "Area" << setw(15) << "Center(X,Y)" << setw(10) << "Kr2" << setw(6) << "Circle\n";
    out << "----------------------------------------------------------------\n";
    for (int l = 1; l <= num_black; ++l) {
        const auto& r = info_black[l];
        if (r.m00 == 0) continue;
        if (r.isCircle) cnt_black++;
        string center_str = "(" + to_string((int)r.xc) + ", " + to_string((int)r.yc) + ")";
    
        out  << r.id 
        << "    " << fixed << setprecision(2) << r.m00 
        << "    " << left << center_str
        << "    " << setw(2) << right << fixed << setprecision(4) << r.kr2 
        << "    " << setw(6) << (r.isCircle ? "YES" : "NO") << "\n";
    }
    out << "Circle-like (Area>30): " << cnt_black << "\n";

    cout << "-> Folder: " << folder << "/" << endl;
    cout << "-> White circles: " << cnt_white << ", Black circles: " << cnt_black << endl;
}

int main() {
    vector<string> files = {"1.png", "2.png", "3.png", "4.png"};
    for (const auto& f : files) processImage(f);
    return 0;
}