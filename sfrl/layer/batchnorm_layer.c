#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "sfrl/activations/activations.h"
#include "sfrl/layer/base_layer.h"
#include "sfrl/layer/batchnorm_layer.h"
#include "sfrl/optimizer/optimizer.h"
#include "sfrl/utils/blas.h"

BatchNormLayer MakeBatchNormLayer(int batch_size, int input_size, int output_size,
                                  ActiType acti_type, InitType init_type) {
  BatchNormLayer layer = {0};
  layer.layer_type = BATCHNORMALIZATION;
  layer.batch_size = batch_size;
  layer.input_size = input_size;
  layer.output_size = output_size;

  /**
   *  bn 层有两个需要学习的参数，γ β，这里 β 其实可以复用bias，但是会造成代码阅读困难
   *  本身beta也没有多少空间占用，就还是给一个单独的参数了
   * */
  layer.bn_gammas = calloc(output_size, sizeof(float));
  layer.bn_gamma_grads = calloc(output_size, sizeof(float));
  layer.bn_betas = calloc(output_size, sizeof(float));
  layer.bn_beta_grads = calloc(output_size, sizeof(float));

  for (i = 0; i < output_size; ++i) {
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
  layer.update = UpdateBatchNormLayer;

  return layer;
}

void ForwardBatchNormLayer(BatchNormLayer *layer, NetWork *net) {
  assert(layer->rolling_momentum > 0);
  float momentum = layer->rolling_momentum;
  if (net.mode == TRIAN) {
    MeanTensor(layer->output, layer->output_size, layer->batch_size, layer->mean);
    VarianceTensor(layer->output, layer->output_size, layer->batch_size, layer->mean,
                   layer->variance);
    /**
     *  计算 norm 并且 存储norm之前的值
     * */
    memcpy(layer->output, layer->output_before_norm, layer->output_size * sizeof(float));
    NormTensor(layer->output, layer->output_size, layer->batch_size, layer->mean, layer->variance);
    memcpy(layer->output, layer->output_normed, layer->output_size * sizeof(float));

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
    NormTensor(layer->output, layer->output_size, layer->batch_size, layer->rolling_mean,
               layer->rolling_variance);
  }
  /**
   *  w*γ +β
   **/
  BatchNormTensor(layer->output, layer->output_size, layer->batch_size, layer->bn_gammas,
                  layer->bn_betas)
}
void BackwardBatchNormLayer(BatchNormLayer *layer, NetWork *net) {
  if (net.mode == TRIAN) {
    /**
     *  求 gamma 和 beta 的梯度
     **/
    BnGamaBackward(layer->bn_gamma_grads, layer->delta, layer->output_normed, layer->output_size,
                   layer->batch_size);
    BnBetaBackward(layer->bn_beta_grads, layer->delta, layer->output_size, layer->batch_size);
    /**
     *  更新delta
     *  1 delta = gamma * delta
     *  2 delta_mean = meand(delta)
     *  3 delta_var = variance(delta)
     *  4 normalize delta
     **/
    BnDot(layer->bn_gammas, layer->delta, layer->output_size, layer->batch_size);
  }
}
void UpdateBatchNormLayer(BatchNormLayer *layer, NetWork *net);

void BnGamaBackward(float *gamma_grads, float *delta, float *output_normed, int input_size,
                    int batch_size) {
  for (int i = 0; i < layer->output_size; ++i) {
    for (int j = 0; j<layer->batch_size; ++j>) {
      gamma_grads[i] += delta[i + input_size * j] * output_normed[i + input_size * j];
    }
  }
}

void BnBetaBackward(float *beta_grads, float *delta, int input_size, int batch_size) {
  for (int i = 0; i < layer->output_size; ++i) {
    for (int j = 0; j<layer->batch_size; ++j>) {
      beta_grads[i] += delta[i + input_size * j];
    }
  }
}

void BnDot(float *gamma, float *delta, int input_size, int batch_size) {
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      delta[i + input_size * j] *= gamma[i];
    }
  }
}

void BnMeanDelta(float *variance, float *delta, int input_size, int batch_size, float *mean) {
  for (int i = 0; i < input_size; ++i) {
    for (int j = 0; j < batch_size; ++j) {
      mean[i] += delta[i + input_size * j];
      delta[i + input_size * j] *= gamma[i];
    }
    mean[i] *= 1 / sqrt(variance[i] + .00001f);
  }
}