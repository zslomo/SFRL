
#ifndef GEMM_H
#define GEMM_H

void AddDot4x4( int, double *, int, double *, int, double *, int );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );
void InnerKernel( int, int, int, double *, int, double *, int, double *, int, int );
void Gemm(int m, int n, int k, double *a, int lda, double *b, int ldb,
          double *c, int ldc);

#endif