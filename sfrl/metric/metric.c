#include "metric.h"
#include <assert.h>
#include <stdlib.h>

float MseMetric(int n, float *pred, float *truth) {
  float error = 0;
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    error += diff * diff;
  }
  return error / n;
}

float AccMetric(int n, int class_num, float *pred, float *truth) {
  float acc_cnt = 0;
  int pred_class = -1;
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    float pred_max = 0;
    for (int j = 0; j < class_num; ++j) {
      // printf("pred: %f ", pred[i * class_num + j]);
      if (pred[i * class_num + j] > pred_max) {
        pred_max = pred[i * class_num + j];
        pred_class = j;
      }
      // printf(", %d", pred_class);
    }
    // printf("\n");
    acc_cnt += (pred_class == (int)truth[i] ? 1 : 0);
  }
  return acc_cnt / n;
}
