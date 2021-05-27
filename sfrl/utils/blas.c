#include "sfrl/utils/blas.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

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
void Gemm(int TransA, int TransB, int M, int N, int K, float ALPHA, float BETA, float *A, int lda,
          float *B, int ldb, float *C, int ldc) {
  // 首先计算BETA*C 这里不涉及转置
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * ldc + j] *= BETA;
    }
  }

  // 判断4种转置，分别调用四种函数
  if (!TransA && !TransB) {
    GemmAB(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  } else if (TransA && !TransB) {
    GemmTAB(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  } else if (!TransA && TransB) {
    GemmATB(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  } else {
    GemmTATB(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  }
}
/**
 *   下面四个函数就是AB是否转置的矩阵乘法
 *   矩阵乘法在没有经过优化的时候就是一个简单的3重for循环
 **/
void GemmAB(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C,
            int ldc) {
  // i表示A的第i行，也是C的第i行
  for (int i = 0; i < M; ++i) {
    // k表示A的第k列，同时表示B的第k行
    for (int k = 0; k < K; ++k) {
      // ALPHA * A 利用 register关键字可以让编译器吧这部分数据缓存在cpu寄存器中，加速计算的手段
      register float A_PART = ALPHA * A[i * lda + k];
      // j是B的第j列，也是C的第j列
      for (int j = 0; j < N; ++j) {
        // A中第i行k列与B所有列第k行所有元素相乘的结果
        // 这里多说几句，分页调度是计算机组成的经典问题，按什么样的顺序访问矩阵显然会影响性能
        // CSAPP的封面讲的就是正确和错误的访问顺序有多大的性能差距，而且先行后列的访问是正确的
        // 矩阵乘法中需要用A 的行去乘 B的列，A没有问题，B就有问题了，B是先列后行，所以这里先把
        // A计算好的部分给regist掉，然后按行访问B，这就解决了这个问题
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void GemmTAB(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C,
             int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      register float A_PART = ALPHA * A[k * lda + i];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void GemmATB(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C,
             int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      register float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
}

void GemmTATB(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C,
              int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      register float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
      }
      C[i * ldc + j] += sum;
    }
  }
}

// 一维点积
float DotSumTensor(int size, float *TensorX, float *TensorY) {
  float dot = 0;
  for (int i = 0; i < size; ++i) {
    dot += TensorX[i] * TensorY[i];
  }
  return dot;
}

void DotTensor(int size, float *TensorX, float *TensorY) {
  float dot = 0;
  for (int i = 0; i < size; ++i) {
    TensorY[i] = TensorX[i] * TensorY[i];
  }
}

void SquareTensor(int size, float *TensorX, float *TensorY) {
  for (int i = 0; i < size; ++i) {
    TensorY[i] = TensorX[i] * TensorX[i];
  }
}

void SqrtTensor(int size, float *TensorX, float *TensorY) {
  for (int i = 0; i < size; ++i) {
    TensorY[i] = sqrt(TensorX[i]);
  }
}

void FillTensorBySingleValue(int size, float *Tensor, float value) {
  for (int i = 0; i < size; ++i) {
    Tensor[i] = value;
  }
}

void AxpyTensor(int size, float ALPHA, float *TensorX, float *TensorY) {
  for (int i = 0; i < size; ++i) {
    TensorY[i] += ALPHA * TensorX[i];
  }
}

void ScalTensor(int size, float ALPHA, float *TensorX) {
  for (int i = 0; i < size; ++i) {
    TensorX[i] *= ALPHA;
  }
}

void DivTensor(int size, float eps, float ALPHA, float *TensorX, float *TensorY) {
  for (int i = 0; i < size; ++i) {
    TensorY[i] = ALPHA / (TensorX[i] + eps);
  }
}

void MeanTensor(float *TensorX, int input_size, int batch_size, float *mean) {
  assert(batch_size > 1);
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      mean[i] += TensorX[i + input_size * j];
    }
    mean[i] *= 1 / batch_size;
  }
}

void VarianceTensor(float *TensorX, int input_size, int batch_size, float *mean, float *variance) {
  assert(batch_size > 1);
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      variance[i] += pow(TensorX[i + input_size * j] - mean[i], 2);
    }
    variance[i] *= 1 / (batch_size - 1);
  }
}

void NormTensor(float *TensorX, int input_size, int batch_size, float *mean, float *variance) {
  assert(batch_size > 1);
  float eps = 1e-8;
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      TensorX[i + input_size * j] = (TensorX[i + input_size * j] - mean[i]) / (sqrt(variance[i]));
    }
  }
}

void BatchNormTensor(float *TensorX, int input_size, int batch_size, float *gamma, float *beta) {
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      TensorX[i + input_size * j] = TensorX[i + input_size * j] * gamma[i] + beta[i];
    }
  }
}

void InitTensor(int size, float ALPHA, float *TensorX) {
  for (int i = 0; i < size; ++i) {
    TensorX[i] = ALPHA;
  }
}