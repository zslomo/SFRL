
#include "loss.h"
#include "../../sfrl/utils/blas.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

// MSE 这里不求和
void MeanSquareError(int n, float *pred, float *truth, float *error) {
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    error[i] = diff * diff / n;
  }
}

void BackwardMeanSquareError(int n, float *pred, float *truth, float *delta) {
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    // delta的 每一项可能非常小，/n 会弥散，
    delta[i] = diff;
  }
}

void CrossEntropy(int batch_size, int class_num, float *pred, float *truth, float *error,
                  int weight_ce) {
  assert(pred);
  assert(truth);
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < class_num; ++j) {
      // weight_ce 是强化学习的特殊形式，强化学习没有监督信号，
      // truth并不是一个类别标签 而是权重
      int t = weight_ce ? truth[i] : (int)truth[i] & j;
      error[i] += -t * log(pred[j * i + j]);
    }
  }
}

void BackwardCrossEntropy(int batch_size, int class_num, float *pred, float *truth, float *delta,
                          int weight_ce) {
  assert(pred);
  assert(truth);
  float eps = 1e-8;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < class_num; ++j) {
      int t = weight_ce ? truth[i] : (int)truth[i] & j;
      delta[i] = t - pred[j * i + j];
    }
  }
}