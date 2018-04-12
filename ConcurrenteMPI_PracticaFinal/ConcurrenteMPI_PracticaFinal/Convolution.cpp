#include "Convolution.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
using namespace std;

#define CLAMP(a, b, c) std::min(std::max(a, b), c)
#define CELL_POS(x, Mx, y, My) (CLAMP(x, 0, Mx-1) + Mx * CLAMP(y, 0, My-1))

float applyFilter(float* matrix, int mWidth, int mHeight,
                  float* filter, int fWidth, int fHeight,
                  int cellX, int cellY) {

    float sum = 0.0f;
    int fwr = fWidth / 2;
    int fhr = fHeight / 2;
    int x = cellX - fwr;
    int y = cellY - fhr;
    float currValue;

    for (int i = 0; i < fHeight; ++i) {
        for (int j = 0; j < fWidth; ++j) {
            int pos = CELL_POS((x + i), mHeight, (y + j), mWidth);
            currValue = matrix[pos];
            sum += currValue * filter[i + j * fWidth];
        }
    }
    return sum;
}

float* makeConvolution( float* matrix, int mWidth, int mHeight,
                        float* filter, int fWidth, int fHeight) {

    float filterSum = 0.0f;
    float* data = new float[mWidth*mHeight];
    for (int i = 0; i < fHeight; ++i) {
        for (int j = 0; j < fWidth; ++j) {
            filterSum += filter[i + j * fHeight];
        }
    }

    int k = 0;
    #pragma omp parallel for
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < mWidth; ++j) {
            data[CELL_POS(i, mHeight, j, mWidth)] = applyFilter(matrix, mWidth, mHeight, filter, fWidth, fHeight, i, j) / filterSum;
        }
        stringstream ss;
        ss << ((float)(k++ * 100) / mHeight) << "%\r";
        cout << ss.str();
    }


    return data;
}