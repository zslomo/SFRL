#include "base_layer.h"
#include "../network/network.h"
#include "../optimizer/optimizer.h"
#include "../utils/blas.h"
#include <stdio.h>
#include <stdlib.h>

void UpdateLayer(Layer *layer, Network *net) {
  int input_size = layer->input_size;
  int output_size = layer->output_size;
  int batch_size = net->batch_size;
  int w_size = input_size * output_size;
  int b_size = output_size;
  float *weights = layer->weights;
  float *weight_grads = layer->weight_grads;
  float *biases = layer->biases;
  float *bias_grads = layer->bias_grads;
  if (!layer->grad_cum_w) {
    layer->grad_cum_w = calloc(w_size, sizeof(float));
  }
  float *grad_cum_w = layer->grad_cum_w;
  if (!layer->grad_cum_b) {
    layer->grad_cum_b = calloc(b_size, sizeof(float));
  }
  float *grad_cum_b = layer->grad_cum_b;
  if (!layer->grad_cum_square_w) {
    layer->grad_cum_square_w = calloc(w_size, sizeof(float));
  }
  float *grad_cum_square_w = layer->grad_cum_square_w;
  if (!layer->grad_cum_square_b) {
    layer->grad_cum_square_b = calloc(b_size, sizeof(float));
  }
  float *grad_cum_square_b = layer->grad_cum_square_b;

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

void ResetLayer(Layer *layer) {
  if (layer->input) {
    InitTensor(layer->input_size * layer->batch_size, 0, layer->input);
  }
  if (layer->output) {
    InitTensor(layer->output_size * layer->batch_size, 0, layer->output);
  }
  if (layer->delta) {
    InitTensor(layer->output_size * layer->batch_size, 0, layer->delta);
  }
  if (layer->weight_grads) {
    InitTensor(layer->output_size * layer->input_size, 0, layer->output);
  }
  if (layer->bias_grads) {
    InitTensor(layer->output_size * layer->input_size, 0, layer->output);
  }
  if (layer->bn_gamma_grads) {
    InitTensor(layer->output_size, 0, layer->bn_gamma_grads);
  }
  if (layer->bn_beta_grads) {
    InitTensor(layer->output_size, 0, layer->bn_beta_grads);
  }
  if (layer->output_normed) {
    InitTensor(layer->output_size * layer->input_size, 0, layer->output_normed);
  }
  if (layer->output_before_norm) {
    InitTensor(layer->output_size * layer->input_size, 0, layer->output_before_norm);
  }
}

void PrintWeight(Layer *layer) {
  int n = layer->input_size;
  int m = layer->output_size;
  int batch_size = layer->batch_size;
  if (layer->layer_name) {
    printf("layer %s weight size = %d × %d + %d\n", layer->layer_name, n, m, m);
  } else {
    printf("layer %s weight size = %d × %d + %d\n", GetLayerTypeStr(layer->layer_type), n, m, m);
  }

  printf("bias:\n");
  for (int i = 0; i < m; ++i) {
    printf("%f ", layer->biases[i]);
  }
  printf("\n");
  printf("weight :\n");
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", layer->weights[n * i + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void PrintInput(Layer *layer, int batch_num) {
  int n = layer->input_size;
  int batch_size = layer->batch_size;
  batch_num = batch_num > batch_size ? batch_size : batch_num;
  if (layer->layer_name) {
    printf("layer %s input size = %d × %d\n", layer->layer_name, batch_size, n);
  } else {
    printf("layer %s input size = %d × %d\n", GetLayerTypeStr(layer->layer_type), batch_size, n);
  }
  printf("inputs:\n");

  for (int i = 0; i < batch_num; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", layer->input[n * i + j]);
    }
    printf("\n");
  }
}

void PrintOutput(Layer *layer, int batch_num) {
  int n = layer->output_size;
  int batch_size = layer->batch_size;
  batch_num = batch_num > batch_size ? batch_size : batch_num;
  if (layer->layer_name) {
    printf("layer %s output size = %d × %d\n", layer->layer_name, batch_size, n);
  } else {
    printf("layer %s output size = %d × %d\n", GetLayerTypeStr(layer->layer_type), batch_size, n);
  }
  printf("outputs:\n");

  for (int i = 0; i < batch_num; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", layer->output[n * i + j]);
    }
    printf("\n");
  }
}

void PrintGrad(Layer *layer) {
  int n = layer->input_size;
  int m = layer->output_size;
  if (layer->layer_name) {
    printf("layer %s weight size = %d × %d + %d\n", layer->layer_name, n, m, m);
  } else {
    printf("layer %s weight size = %d × %d + %d\n", GetLayerTypeStr(layer->layer_type), n, m, m);
  }
  printf("bias grad:\n");
  for (int i = 0; i < m; ++i) {
    printf("%f ", layer->bias_grads[i]);
  }
  printf("\n");
  printf("weight grad:\n");
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", layer->weight_grads[n * i + j]);
    }
    printf("\n");
  }
}

void PrintDelta(Layer *layer, int batch_num) {
  int n = layer->output_size;
  int batch_size = layer->batch_size;
  batch_num = batch_num > batch_size ? batch_size : batch_num;
  if (layer->layer_name) {
    printf("layer %s delta size = %d × %d\n", layer->layer_name, batch_size, n);
  } else {
    printf("layer %s delta size = %d × %d\n", GetLayerTypeStr(layer->layer_type), batch_size, n);
  }
  printf("delta:\n");

  for (int i = 0; i < batch_num; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", layer->delta[n * i + j]);
    }
    printf("\n");
  }
}

char *GetLayerTypeStr(LayerType layer_type) {
  char *layer_type_str;
  if (layer_type == DENSE) {
    layer_type_str = "Dense";
  } else if (layer_type == BATCHNORMALIZATION) {
    layer_type_str = "BatchNorm";
  } else if (layer_type == SOFTMAX) {
    layer_type_str = "SoftMax";
  } else if (layer_type == DROPOUT) {
    layer_type_str = "DropOut";
  } else if (layer_type == ACTIVATION) {
    layer_type_str = "Activation";
  } else if (layer_type == LOSS) {
    layer_type_str = "Loss";
  } else {
    layer_type_str = "error";
  }
  return layer_type_str;
}