
#include "sfrl/loss/loss.h"
#include "sfrl/utils/blas.h"
#include <assert.h>
#include <float.h>
#include <math.h>

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

void CrossEntropy(int n, float *pred, float *truth, float *error) {
  assert(pred);
  assert(truth);
  for (int i = 0; i < n; ++i) {
    error[i] = truth[i] * log(pred[i]);
  }
}

void BackwardCrossEntropy(int n, float *pred, float *truth, float *delta) {
  assert(pred);
  assert(truth);
  float eps = 1e-8;
  for (int i = 0; i < n; ++i) {
    delta[i] = truth[i] / (log(pred[i]) + eps) / n;
  }
}

// 加权 交叉熵，policy gradient 会使用
void CrossEntropyWithWeight(int n, float *pred, float *weight, float *error) {
  assert(pred);
  assert(weight);
  for (int i = 0; i < n; ++i) {
    error[i] = weight[i] * log(pred[i]);
  }
}

void BackwardCrossEntropyWithWeight(int n, float *pred, float *weight, float *delta) {
  assert(pred);
  assert(weight);
  float eps = 1e-8;
  for (int i = 0; i < n; ++i) {
    delta[i] = weight[i] / (log(pred[i]) + eps) / n;
  }
}
