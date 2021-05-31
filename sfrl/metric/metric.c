#include <assert.h>
#include <stdlib.h>
#include "sfrl/metric/metric.h"

float MseMetric(int n, float *pred, float *truth) {
  float error = 0;
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    error += diff * diff;
  }
  return error / n;
}

float AccMetric(int n, float threshold, float *pred, float *truth) {
  float acc_cnt = 0;
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    acc_cnt += (pred[i] >= threshold && truth[i] == 1.0) || (pred[i] < threshold && truth[i] == 0.0)
                   ? 1
                   : 0;
  }
  return acc_cnt / n;
}
