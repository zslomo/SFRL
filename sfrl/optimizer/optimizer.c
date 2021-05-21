#include "sfrl/optimizer/optimizer.h"

#include <math.h>

/**
 * Vt = γ * Vt-1 + lr * dWt
 * Wt = Wt-1 - Vt
 * γ : momentum
 **/
void SgdOptimizer(int input_size, int output_size, int batch_size, float *weights,
                  float *weight_updates, float *biases, float *bias_updates, float lr, float decay,
                  float momentum) {
  // bias
  AxpyTensor(output_size, -lr, bias_updates, biases);
  ScalTensor(output_size, momentum, bias_updates);

  // weight
  AxpyTensor(input_size * output_size, decay, weights, weight_updates);
  AxpyTensor(input_size * output_size, -lr, weight_updates, weights);
  // 动量更新
  ScalTensor(input_size * output_size, momentum, weight_updates);
}