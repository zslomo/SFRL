#ifndef BLAS_H
#define BLAS_H


/**
 *  初级的gemm算法，没有经过4×4加速，C = ALPHA * A * B + BETA * C
 *  参数：
 *      A B C 三个矩阵
 *      TransA TransB 是否转置
 *      M A的行数，实际就是C的行数
 *      N B的列数，实际就是C的列数
 *      K A的列数，B的行数
 *      ALPHA， BETA 系数
 *      lda A的列数，转制后是A的行数
 *      ldb B的列数，转制后是B的行数
 *      ldc C的列数
 *  一些说明：
 *    转置这里，后面可以看到分了4个函数实现转置与否的四种情况，本来打算实现一个转置函数，但是这就是两步操作，显然会变慢
 *    ld(abc)
 *三个参数是用来找到flatten后的元素位置的，矩阵都会转为一维存储，按行flat
 *    理论上来说，用4*4的kernel来优化访存可以获得8倍的速度提升，一般也都会这样做，对应的代码在gemm.c中有，只是比较复杂
 *    实际用的时候要考虑矩阵过小等情况下怎么退化成一般矩阵乘法、平台cpu缓存有多大，到底用多大的kernel合适等等问题
 *    目前没有精力折腾，当成一个todo吧
 *    具体的GEMM优化方法可以参见 https://zhuanlan.zhihu.com/p/66958390
 **/

void Gemm(int TransA, int TransB, int M, int N, int K, float ALPHA, float BETA,
          float *A, int lda, float *B, int ldb, float *C, int ldc);
void GemmAB(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
            int ldb, float *C, int ldc);
void GemmTAB(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
             int ldb, float *C, int ldc);
void GemmATB(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
             int ldb, float *C, int ldc);
void GemmTATB(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
              int ldb, float *C, int ldc);

float dotd1(int N, float *X, int INCX, float *Y, int INCY);
#endif