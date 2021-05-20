#include "sfrl/utils/metrics.h"
#include <math.h>

float RootMeanSquareError(int n, float *pred, float *truth) {
  assert(n > 0);
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    sum += diff * diff / n;
  }
  return sqrt(sum);
}