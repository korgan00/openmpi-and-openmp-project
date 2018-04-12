#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>
#include "Convolution.h"
using namespace std;

#define MATRIX_SIZE 10000

void generateMatrix(int width, int height, float* &data);
void generateMatrixFile(int width, int height);
void readMatrixFile(int& w, int& h, float* &data);

int main() {
    //generateMatrixFile(MATRIX_SIZE, MATRIX_SIZE);

    cout << endl << endl;

    int w, h;
    float* mat = nullptr;
    //readMatrixFile(w, h, mat);
    auto start = std::chrono::system_clock::now();

    generateMatrix(w = MATRIX_SIZE, h = MATRIX_SIZE, mat);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count() << " s" << endl;

    float filter[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f }
    };



#if MATRIX_SIZE <= 15
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            cout << mat[i * w + j] << " ";
        }
        cout << endl;
    }
#endif

    cout << endl;
    cout << endl;
    
    cout << "started" << endl;
    start = std::chrono::system_clock::now();

    float* result = makeConvolution(mat, w, h, &filter[0][0], 3, 3);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << diff.count() << " s" << endl;

#if MATRIX_SIZE <= 15
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            cout << result[i * w + j] << " ";
        }
        cout << endl;
    }
#endif

    std::cin.get();
    return 0;
}

void readMatrixFile(int& w, int& h, float* &data) {
    ifstream ifs;
    stringstream ss;
    ss << "../matrix/matrix" << MATRIX_SIZE << ".bin";

    ifs.open(ss.str().c_str(), ofstream::in | ofstream::binary);
    if (!ifs.is_open()) {
        cerr << "ERROR: CANNOT OPEN READ FILE";
    }
    float n;

    ifs >> w;
    ifs >> h;
    cout << w << endl;
    cout << h << endl;

    data = new float[w*h];

    cout << "Reading matrix" << endl;
    int k = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            ifs >> n;
            data[k++] = n;
        }
        cout << ((float)(i*100)/h) << "%\r";
    }
    ifs.close();
    cout << "100%             " << endl;
    cout << "Matrix read" << endl;
}


void generateMatrixFile(int width, int height) {
    ofstream ofs;
    stringstream ss;
    ss << "../matrix/matrix" << width << ".bin";
    ofs.open(ss.str().c_str(), ofstream::out | ofstream::binary | ofstream::trunc);

    ofs << width << endl;
    ofs << height << endl;

    cout << "Generating matrix" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            ofs << (rand() % 1000) << endl;
            //if (i % MATRIX_SIZE == 0) cout << (i / MATRIX_SIZE) << endl;
        }
        cout << ((float)(i * 100) / height) << "%\r";
    }
    ofs.close();
    cout << "100%               " << endl;
    cout << "Matrix generated" << endl;
}

void generateMatrix(int w, int h, float* &data) {

    data = new float[w*h];

    cout << "Generating matrix" << endl;
    long int k = 0, l = 0;
    #pragma omp parallel for
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            data[k++] = (float)(rand() % 1000);
        }
        stringstream ss;
        ss << ((float)(l++ * 100) / h) << "%\r";
        cout << ss.str();
    }
    cout << "100%               " << endl;
    cout << "Matrix generated" << endl;
}
