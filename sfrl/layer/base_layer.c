#include "sfrl/layer/layer.h"
#include "sfrl/network/network.h"

void UpdateLayer(Layer *layer, NetWork *network) {
  int input_size = layer->input_size;
  int output_size = layer->output_size;
  int batch_size = network->batch_size;
  float *weights = layer->weights;
  float *weight_grads = layer->weight_grads;
  float *biases = layer->biases;
  float *bias_grads = layer->bias_grads;
  float *grad_cum_w = network->grad_cum_w;
  float *grad_cum_b = network->grad_cum_b;
  float *grad_cum_square_w = network->grad_cum_square_w;
  float *grad_cum_square_b = network->grad_cum_square_b;
  float lr = network->learning_rate;
  float beta_1 = network->beta_1;
  float beta_2 = network->beta_2;
  float momentum = network->momentum;
  float decay = network->decay;

  switch (layer->acti_type) {
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

void FreeLayer(Layer layer) {
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

  if (layer->scales) {
    free(layer->scales);
  }
  if (layer->scale_updates) {
    free(layer->scale_updates);
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
  if (layer->norm_input) {
    free(layer->norm_input);
  }
  if (layer->norm_output) {
    free(layer->norm_output);
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