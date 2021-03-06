#include "optimizer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../utils/blas.h"

void SgdOptimizer(Network *net, Layer *layer) {
  int input_size = layer->input_size;
  int output_size = layer->output_size;
  int batch_size = net->batch_size;
  int w_size = input_size * output_size;
  int b_size = output_size;
  float *weights = layer->weights;
  float *weight_grads = layer->weight_grads;
  float *biases = layer->biases;
  float *bias_grads = layer->bias_grads;
  float *w_updates = layer->weight_updates;
  float *b_updates = layer->bias_updates;
  float *grad_cum_b = layer->grad_cum_b;
  float *grad_cum_w = layer->grad_cum_w;
  float lr = net->learning_rate;
  float momentum = net->momentum;
  float decay = net->decay;

  // bias
  float *grad_tmp_b = calloc(b_size, sizeof(float));
  memcpy(grad_tmp_b, grad_cum_b, b_size * sizeof(float));
  // ρVt-1
  ScalTensor(b_size, 1 - momentum, grad_tmp_b);
  // g + ρVt-1
  AxpyTensor(b_size, 1, bias_grads, grad_tmp_b);
  // b = b - lr * (g + ρVt-1)
  AxpyTensor(b_size, -lr / batch_size, grad_tmp_b, biases);
  AxpyTensor(b_size, -lr / batch_size, grad_tmp_b, b_updates);
  // 动量更新d
  memcpy(grad_cum_b, grad_tmp_b, b_size * sizeof(float));
  free(grad_tmp_b);

  // weight
  float *grad_tmp_w = calloc(w_size, sizeof(float));
  memcpy(grad_tmp_w, grad_cum_w, w_size * sizeof(float));
  // ρVt-1
  ScalTensor(w_size, 1 - momentum, grad_tmp_w);
  // g + ρVt-1
  AxpyTensor(w_size, 1, weight_grads, grad_tmp_w);
  // w = w - lr * (g + ρVt-1)
  AxpyTensor(w_size, -lr / batch_size, grad_tmp_w, weights);
  AxpyTensor(w_size, -lr / batch_size, grad_tmp_w, w_updates);
  // 动量更新
  memcpy(grad_cum_w, grad_tmp_w, w_size * sizeof(float));
  free(grad_tmp_w);
}

void AdaGradOptimizer(Network *net, Layer *layer) {
  int input_size = layer->input_size;
  int output_size = layer->output_size;
  int batch_size = net->batch_size;
  int w_size = input_size * output_size;
  int b_size = output_size;
  float *weights = layer->weights;
  float *weight_grads = layer->weight_grads;
  float *biases = layer->biases;
  float *bias_grads = layer->bias_grads;
  float *w_updates = layer->weight_updates;
  float *b_updates = layer->bias_updates;
  float *grad_cum_b = layer->grad_cum_b;
  float *grad_cum_w = layer->grad_cum_w;
  float *grad_cum_square_b = layer->grad_cum_square_b;
  float *grad_cum_square_w = layer->grad_cum_square_w;
  float lr = net->learning_rate;
  float beta_1 = net->beta_1;
  float beta_2 = net->beta_2;
  float momentum = net->momentum;
  float decay = net->decay;

  float eps = 1e-4;

  // 梯度累计量计算
  // r = r + g*g

  assert(grad_cum_square_b);
  assert(grad_cum_square_w);

  // Rt = Rt-1 + g*g
  float *grad_tmp_b = calloc(b_size, sizeof(float));
  SquareTensor(b_size, bias_grads, grad_tmp_b);
  AxpyTensor(b_size, 1, grad_tmp_b, grad_cum_square_b);
  free(grad_tmp_b);

  float *grad_tmp_w = calloc(w_size, sizeof(float));
  SquareTensor(w_size, weight_grads, grad_tmp_w);
  AxpyTensor(w_size, 1, grad_tmp_w, grad_cum_square_w);
  free(grad_tmp_w);

  // bias
  float *increment_tmp_b = calloc(b_size, sizeof(float));
  DivTensor(b_size, eps, 1, grad_cum_square_b, increment_tmp_b);
  AxpyTensor(b_size, -lr / batch_size, increment_tmp_b, biases);
  AxpyTensor(b_size, -lr / batch_size, increment_tmp_b, b_updates);
  free(increment_tmp_b);
  // weight
  float *increment_tmp_w = calloc(w_size, sizeof(float));
  DivTensor(w_size, eps, 1, grad_cum_square_w, increment_tmp_w);
  AxpyTensor(w_size, -lr / batch_size, increment_tmp_w, weights);
  AxpyTensor(w_size, -lr / batch_size, increment_tmp_w, w_updates);
  free(increment_tmp_w);
}

void RmsPropOptimizer(Network *net, Layer *layer) {
  int input_size = layer->input_size;
  int output_size = layer->output_size;
  int batch_size = net->batch_size;
  int w_size = input_size * output_size;
  int b_size = output_size;
  float *weights = layer->weights;
  float *weight_grads = layer->weight_grads;
  float *biases = layer->biases;
  float *bias_grads = layer->bias_grads;
  float *w_updates = layer->weight_updates;
  float *b_updates = layer->bias_updates;
  float *grad_cum_b = layer->grad_cum_b;
  float *grad_cum_w = layer->grad_cum_w;
  float *grad_cum_square_b = layer->grad_cum_square_b;
  float *grad_cum_square_w = layer->grad_cum_square_w;
  float lr = net->learning_rate;
  float beta_1 = net->beta_1;
  float beta_2 = net->beta_2;
  float momentum = net->momentum;
  float decay = net->decay;
  /**
   *  Rmsprop 中存在一个1/r
   * 的对梯度的衰减，r是累计平方梯度，当激活函数是relu的时候
   *  梯读会出现大量的0(输入为负梯度就是0) 那么 1/r
   * 就会爆炸，所以这里eps不宜设置过大 否则 衰减系数太大，不容易收敛
   * */
  float eps = 1e-4;

  assert(grad_cum_square_b);
  assert(grad_cum_square_w);

  // Rt = ρ*Rt-1 + (1-ρ)*g*g
  float *grad_tmp_b = calloc(b_size, sizeof(float));
  // g*g
  SquareTensor(b_size, bias_grads, grad_tmp_b);
  // (1-ρ)*g*g
  ScalTensor(b_size, 1 - decay, grad_tmp_b);
  // ρ*Rt-1
  ScalTensor(b_size, decay, grad_cum_square_b);
  // ρ*Rt-1 + (1-ρ)*g*g
  AxpyTensor(b_size, 1, grad_tmp_b, grad_cum_square_b);
  free(grad_tmp_b);
  float *grad_tmp_w = calloc(w_size, sizeof(float));
  // g*g
  SquareTensor(w_size, weight_grads, grad_tmp_w);
  // (1-ρ)*g*g
  ScalTensor(w_size, 1 - decay, grad_tmp_w);
  // ρ*Rt-1
  ScalTensor(w_size, decay, grad_cum_square_w);
  // ρ*Rt-1 + (1-ρ)*g*g
  AxpyTensor(w_size, 1, grad_tmp_w, grad_cum_square_w);
  free(grad_tmp_w);

  // bias
  // b = b - lr * 1 / sqrt(r + eps)
  float *increment_tmp_b = calloc(b_size, sizeof(float));
  // r + eps
  AxpyTensor(b_size, eps, grad_cum_square_b, increment_tmp_b);
  // sqrt(r + eps)
  SqrtTensor(b_size, increment_tmp_b, increment_tmp_b);
  // 1 / sqrt(r + eps)
  DivTensor(b_size, eps, 1, increment_tmp_b, increment_tmp_b);
  // b - lr * 1 / sqrt(r + eps)
  AxpyTensor(b_size, -lr / batch_size, increment_tmp_b, biases);
  AxpyTensor(b_size, -lr / batch_size, increment_tmp_b, b_updates);
  free(increment_tmp_b);

  // weight
  float *increment_tmp_w = calloc(w_size, sizeof(float));
  AxpyTensor(w_size, eps, grad_cum_square_w, increment_tmp_w);
  SqrtTensor(w_size, increment_tmp_w, increment_tmp_w);
  DivTensor(w_size, eps, 1, increment_tmp_w, increment_tmp_w);
  AxpyTensor(w_size, -lr / batch_size, increment_tmp_w, weights);
  AxpyTensor(w_size, -lr / batch_size, increment_tmp_w, w_updates);
  free(increment_tmp_w);
}

void AdamOptimizer(Network *net, Layer *layer) {
  int input_size = layer->input_size;
  int output_size = layer->output_size;
  int batch_size = net->batch_size;
  int w_size = input_size * output_size;
  int b_size = output_size;
  float *weights = layer->weights;
  float *weight_grads = layer->weight_grads;
  float *biases = layer->biases;
  float *bias_grads = layer->bias_grads;
  float *w_updates = layer->weight_updates;
  float *b_updates = layer->bias_updates;
  float *grad_cum_b = layer->grad_cum_b;
  float *grad_cum_w = layer->grad_cum_w;
  float *grad_cum_square_b = layer->grad_cum_square_b;
  float *grad_cum_square_w = layer->grad_cum_square_w;
  float lr = net->learning_rate;
  float beta_1 = net->beta_1;
  float beta_2 = net->beta_2;
  float momentum = net->momentum;
  float decay = net->decay;

  float *m_hat_b = calloc(b_size, sizeof(float));
  float *m_hat_w = calloc(b_size, sizeof(float));
  float *v_hat_b = calloc(b_size, sizeof(float));
  float *v_hat_w = calloc(b_size, sizeof(float));

  float eps = 1e-4;

  /**
   *  bias 的Mt Vt更新计算
   *  Mt = Mt-1 + (1 - β1) * g
   *  M_hat = Mt / (1 - β1)
   *  Vt = Vt-1 + (1 - β1) * g*g
   *  V_hat = Bt / (1 - β2)
   **/
  assert(grad_cum_b);
  assert(grad_cum_square_b);

  float *grad_tmp_b = calloc(b_size, sizeof(float));
  // Mt = Mt-1 + (1 - β1) * g
  AxpyTensor(b_size, 1 - beta_1, bias_grads, grad_tmp_b);
  AxpyTensor(b_size, 1, grad_tmp_b, grad_cum_b);
  // M_hat = Mt / (1 - β1)
  ScalTensor(b_size, 1 / (1 - beta_1), m_hat_b);
  // Vt = Vt-1 + (1 - β2) * g*g
  InitTensor(b_size, 0, grad_tmp_b);
  SquareTensor(b_size, bias_grads, grad_tmp_b);
  ScalTensor(b_size, 1 - beta_2, grad_tmp_b);
  AxpyTensor(b_size, 1, grad_tmp_b, grad_cum_square_b);
  // V_hat = Vt / (1 - β2)
  ScalTensor(b_size, 1 / (1 - beta_2), v_hat_b);
  free(grad_tmp_b);
  /**
   *  weight 的Mt Vt更新计算
   *  Mt = Mt-1 + (1 - β1) * g
   *  M_hat = Mt / (1 - β1)
   *  Vt = Vt-1 + (1 - β1) * g*g
   *  V_hat = Bt / (1 - β2)
   **/

  assert(grad_cum_w);
  assert(grad_cum_square_w);
  float *grad_tmp_w = calloc(w_size, sizeof(float));
  // Mt = Mt-1 + (1 - β1) * g
  AxpyTensor(w_size, 1 - beta_1, weight_grads, grad_tmp_w);
  AxpyTensor(w_size, 1, grad_tmp_w, grad_cum_w);
  // M_hat = Mt / (1 - β1)
  ScalTensor(w_size, 1 / (1 - beta_1), m_hat_w);
  // Vt = Vt-1 + (1 - β2) * g*g
  InitTensor(w_size, 0, grad_tmp_w);
  SquareTensor(w_size, weight_grads, grad_tmp_w);
  ScalTensor(w_size, 1 - beta_2, grad_tmp_w);
  AxpyTensor(w_size, 1, grad_tmp_w, grad_cum_square_w);
  // V_hat = Vt / (1 - β2)
  ScalTensor(w_size, 1 / (1 - beta_2), v_hat_w);
  free(grad_tmp_w);

  /**
   * 更新 w = w - lr * m_hat / sqrt(v_hat)
   **/
  // biases
  SqrtTensor(b_size, v_hat_b, v_hat_b);
  DivTensor(b_size, eps, 1, v_hat_b, v_hat_b);
  DotTensor(b_size, m_hat_b, v_hat_b);
  AxpyTensor(b_size, -lr / batch_size, v_hat_b, biases);
  AxpyTensor(b_size, -lr / batch_size, v_hat_b, b_updates);

  // weights
  SqrtTensor(w_size, v_hat_w, v_hat_w);
  DivTensor(w_size, eps, 1, v_hat_w, v_hat_w);
  DotTensor(w_size, m_hat_w, v_hat_w);
  AxpyTensor(w_size, -lr / batch_size, v_hat_w, weights);
  AxpyTensor(w_size, -lr / batch_size, v_hat_w, w_updates);
}