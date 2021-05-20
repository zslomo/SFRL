#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
  int rows, cols; // 矩阵的行与列数
  float **vals;   // 矩阵所存储的数据，二维数组
} Matrix;

Matrix MakeMatrix(int rows, int cols);
void FreeMatrix(Matrix matrix);
Matrix CopyMatrix(Matrix matrix);
Matrix AddMtrix(Matrix m_source, Matrix m_dest);
// 点乘，叉乘调用 blas.c中的 Gemm
Matrix DotMtrix(Matrix m_source, Matrix m_dest);
void PrintMatrix(Matrix matrix);

#endif
