#ifndef MATRIX_H
#define MATRIX_H


typedef struct Matrix{
    int rows, cols;     // 矩阵的行与列数
    float **vals;       // 矩阵所存储的数据，二维数组
} Matrix;

#endif
