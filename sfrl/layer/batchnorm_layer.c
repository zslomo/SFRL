#include "batchnorm_layer.h"
#include "../activation/activation.h"
#include "../optimizer/optimizer.h"
#include "../utils/blas.h"
#include "base_layer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

BatchNormLayer MakeBatchNormLayer(int batch_size, int input_size, ActiType acti_type,
                                  InitType init_type, char *layer_name) {
  BatchNormLayer layer = {0};
  layer.layer_type = BATCHNORMALIZATION;
  layer.layer_name = layer_name;
  layer.batch_size = batch_size;
  layer.input_size = input_size;
  int output_size = input_size;
  layer.output_size = output_size;

  layer.input = calloc(input_size, sizeof(float));
  layer.output = calloc(input_size, sizeof(float));
  // 这里的delta是上一层传承下来的
  layer.delta = calloc(output_size * batch_size, sizeof(float));
  /**
   *  bn 层有两个需要学习的参数，γ β，这里 β 其实可以复用bias，但是会造成代码阅读困难
   *  本身beta也没有多少空间占用，就还是给一个单独的参数了
   * */
  layer.bn_gammas = calloc(output_size, sizeof(float));
  layer.bn_gamma_grads = calloc(output_size, sizeof(float));
  layer.bn_betas = calloc(output_size, sizeof(float));
  layer.bn_beta_grads = calloc(output_size, sizeof(float));

  for (int i = 0; i < output_size; ++i) {
    layer.bn_gammas[i] = 1;
  }

  layer.mean = calloc(output_size, sizeof(float));
  layer.mean_delta = calloc(output_size, sizeof(float));
  layer.variance = calloc(output_size, sizeof(float));
  layer.variance_delta = calloc(output_size, sizeof(float));

  layer.rolling_mean = calloc(output_size, sizeof(float));
  layer.rolling_variance = calloc(output_size, sizeof(float));

  layer.output_normed = calloc(batch_size * output_size, sizeof(float));
  layer.output_before_norm = calloc(batch_size * output_size, sizeof(float));

  layer.forward = ForwardBatchNormLayer;
  layer.backward = BackwardBatchNormLayer;
  layer.print_input = PrintInput;
  layer.print_output = PrintOutput;
  layer.print_delta = PrintDelta;
  layer.reset = ResetLayer;

  return layer;
}

void ForwardBatchNormLayer(BatchNormLayer *layer, Network *net) {
  assert(layer->rolling_momentum > 0);
  float momentum = layer->rolling_momentum;
  CopyTensor(layer->input_size * net->batch_size, net->input, layer->input);
  if (net->mode == TRAIN) {
    MeanTensor(layer->output, layer->output_size, net->batch_size, layer->mean);
    VarianceTensor(layer->output, layer->output_size, net->batch_size, layer->mean,
                   layer->variance);
    /**
     *  计算 norm 并且 存储norm之前的值
     * */
    memcpy(layer->output_before_norm, layer->output, layer->output_size * sizeof(float));
    NormTensor(layer->output, layer->output_size, net->batch_size, layer->mean, layer->variance);
    memcpy(layer->output_normed, layer->output, layer->output_size * sizeof(float));

    /**
     *  计算移动平均 和 移动方差
     * */
    ScalTensor(layer->output_size, momentum, layer->rolling_mean);
    AxpyTensor(layer->output_size, 1 - momentum, layer->rolling_mean, layer->rolling_mean);
    ScalTensor(layer->output_size, momentum, layer->rolling_variance);
    AxpyTensor(layer->output_size, 1 - momentum, layer->rolling_variance, layer->rolling_variance);
  } else {
    /**
     *  Test的时候直接用训练的时候算出来的移动平均和移动方差
     *  详见 https://www.zhihu.com/question/55621104
     * */
    NormTensor(layer->output, layer->output_size, net->batch_size, layer->rolling_mean,
               layer->rolling_variance);
  }
  /**
   *  w*γ +β
   **/
  BatchNormTensor(layer->output, layer->output_size, net->batch_size, layer->bn_gammas,
                  layer->bn_betas);
}
void BackwardBatchNormLayer(BatchNormLayer *layer, Network *net) {

  /**
   *  求 gamma 和 beta 的梯度
   **/
  BnGamaBackward(layer->delta, layer->output_normed, layer->output_size, net->batch_size,
                 layer->bn_gamma_grads);
  BnBetaBackward(layer->delta, layer->output_size, net->batch_size, layer->bn_beta_grads);
  /**
   *  更新delta， 再次说明delta 是对每个加权输入的导数值 dL/dx bn的求导相对复杂，具体公式推导见
   *  https://zhuanlan.zhihu.com/p/45614576
   *  简单的结果公式我写在下面了
   *  1 delta = gamma * delta
   *  2 delta_mean = mean(delta)
   *     1 / sqrt(γ*d)
   *  3 delta_var = variance(delta)
   *     0.5 * sum(d*(obn-mean)) ^ -1.5
   *  4 normalize delta
   *     d = d * 1/(sqrt(v)) + (1 / batch_size) * dm + (2 / batch_size) *dv
   **/
  BnDot(layer->bn_gammas, layer->output_size, net->batch_size, layer->delta);
  BnMeanDelta(layer->variance, layer->delta, layer->bn_gammas, layer->output_size,
              net->batch_size, layer->mean_delta);
  BnNormDelta(layer->output_before_norm, layer->mean, layer->variance, layer->mean_delta,
              layer->variance_delta, layer->input_size, net->batch_size, layer->delta);
  if (net->delta) {
    memcpy(net->delta, layer->delta, layer->output_size * sizeof(float));
  }
}

void BnGamaBackward(float *delta, float *output_normed, int input_size, int batch_size,
                    float *gamma_grads) {
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      gamma_grads[i] += delta[i + input_size * j] * output_normed[i + input_size * j];
    }
  }
}

void BnBetaBackward(float *delta, int input_size, int batch_size, float *beta_grads) {
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      beta_grads[i] += delta[i + input_size * j];
    }
  }
}

void BnDot(float *gamma, int input_size, int batch_size, float *delta) {
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      delta[i + input_size * j] *= gamma[i];
    }
  }
}

void BnMeanDelta(float *variance, float *delta, float *gamma, int input_size, int batch_size,
                 float *mean_delta) {
  float eps = 1e-8;
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      mean_delta[i] += delta[i + input_size * j];
      delta[i + input_size * j] *= gamma[i];
    }
    mean_delta[i] *= 1 / sqrt(variance[i] + eps);
  }
}

void BnVaianceDelta(float *variance_delta, float *output_before_norm, float *delta, float *mean,
                    float *variance, int input_size, int batch_size) {
  float eps = 1e-8;
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      int index = i + input_size * j;
      variance_delta[i] += delta[index] * (output_before_norm[index] - mean[i]);
    }
    variance_delta[i] *= -.5 * pow(variance[i] + eps, (float)(-3. / 2.));
  }
}

void BnNormDelta(float *output_before_norm, float *mean, float *variance, float *mean_delta,
                 float *variance_delta, int input_size, int batch_size, float *delta) {

  float eps = 1e-8;
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      int index = i + input_size * j;
      delta[index] = delta[index] * 1. / (sqrt(variance[i] + index)) +
                     variance_delta[i] * 2. * (output_before_norm[index] - mean[i]) / batch_size +
                     mean_delta[i] / batch_size;
    }
  }
}