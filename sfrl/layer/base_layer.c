#include <stdlib.h>
#include "base_layer.h"
#include "../../sfrl/optimizer/optimizer.h"
#include "../../sfrl/network/network.h"

void UpdateLayer(Layer *layer, NetWork *net) {
  int input_size = layer->input_size;
  int output_size = layer->output_size;
  int batch_size = net->batch_size;
  int w_size = input_size * output_size;
  int b_size = output_size;
  float *weights = layer->weights;
  float *weight_grads = layer->weight_grads;
  float *biases = layer->biases;
  float *bias_grads = layer->bias_grads;
  net->grad_cum_w = calloc(w_size, sizeof(float));
  float *grad_cum_w = net->grad_cum_w;
  net->grad_cum_b = calloc(b_size, sizeof(float));
  float *grad_cum_b = net->grad_cum_b;
  net->grad_cum_square_w = calloc(w_size, sizeof(float));
  float *grad_cum_square_w = net->grad_cum_square_w;
  net->grad_cum_square_b = calloc(b_size, sizeof(float));
  float *grad_cum_square_b = net->grad_cum_square_b;
  float lr = net->learning_rate;
  float beta_1 = net->beta_1;
  float beta_2 = net->beta_2;
  float momentum = net->momentum;
  float decay = net->decay;
  switch (net->opt_type) {
  case ADAM:
    AdamOptimizer(input_size, output_size, weights, weight_grads, biases, bias_grads, grad_cum_w,
                  grad_cum_square_w, grad_cum_b, grad_cum_square_b, lr, beta_1, beta_2);
  case SGD:
    SgdOptimizer(input_size, output_size, weights, weight_grads, biases, bias_grads, grad_cum_w,
                 grad_cum_b, lr, momentum);
  case ADAGRAD:
    AdaGradOptimizer(input_size, output_size, weights, weight_grads, biases, bias_grads,
                     grad_cum_square_w, grad_cum_square_b, lr);
  case RMSPROP:
    RmsPropOptimizer(input_size, output_size, weights, weight_grads, biases, bias_grads,
                     grad_cum_square_w, grad_cum_square_b, lr, decay);
  }
}

void FreeLayer(Layer *layer) {
  if (layer->output) {
    free(layer->output);
  }
  if (layer->delta) {
    free(layer->delta);
  }
  if (layer->weights) {
    free(layer->weights);
  }
  if (layer->weight_grads) {
    free(layer->weight_grads);
  }
  if (layer->biases) {
    free(layer->biases);
  }
  if (layer->bias_grads) {
    free(layer->bias_grads);
  }
  if (layer->bn_gammas) {
    free(layer->bn_gammas);
  }
  if (layer->bn_gamma_grads) {
    free(layer->bn_gamma_grads);
  }
  if (layer->bn_betas) {
    free(layer->bn_betas);
  }
  if (layer->bn_beta_grads) {
    free(layer->bn_beta_grads);
  }
  if (layer->mean) {
    free(layer->mean);
  }
  if (layer->mean_delta) {
    free(layer->mean_delta);
  }
  if (layer->variance) {
    free(layer->variance);
  }
  if (layer->variance_delta) {
    free(layer->variance_delta);
  }
  if (layer->rolling_mean) {
    free(layer->rolling_mean);
  }
  if (layer->rolling_variance) {
    free(layer->rolling_variance);
  }
  if (layer->output_normed) {
    free(layer->output_normed);
  }
  if (layer->output_before_norm) {
    free(layer->output_before_norm);
  }
  if (layer->drop_elem) {
    free(layer->drop_elem);
  }
  if (layer->forward) {
    free(layer->forward);
  }
  if (layer->backward) {
    free(layer->backward);
  }
  if (layer->update) {
    free(layer->update);
  }
}