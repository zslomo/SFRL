#include "base_layer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "../network/network.h"
#include "../optimizer/optimizer.h"
#include "../utils/blas.h"
#include "../utils/utils.h"

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
  float *w_updates = layer->weight_updates;
  float *b_updates = layer->bias_updates;

  float lr = net->learning_rate;
  float beta_1 = net->beta_1;
  float beta_2 = net->beta_2;
  float momentum = net->momentum;
  float decay = net->decay;
  switch (net->opt_type) {
    case ADAM:
      AdamOptimizer(net, layer);
      break;
    case SGD:
      SgdOptimizer(net, layer);
      break;
    case ADAGRAD:
      AdaGradOptimizer(net, layer);
      break;
    case RMSPROP:
      RmsPropOptimizer(net, layer);
      break;
    default:
      break;
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
  // if (layer->input) {
  //   InitTensor(layer->input_size * layer->batch_size, 0, layer->input);
  // }
  // if (layer->output) {
  //   InitTensor(layer->batch_size * layer->output_size, 0, layer->output);
  // }
  if (layer->delta) {
    InitTensor(layer->batch_size * layer->output_size, 0, layer->delta);
  }
  if (layer->weight_grads) {
    InitTensor(layer->input_size * layer->output_size, 0, layer->weight_grads);
  }
  if (layer->bias_grads) {
    InitTensor(layer->output_size, 0, layer->bias_grads);
  }
  if (layer->weight_updates) {
    InitTensor(layer->input_size * layer->output_size, 0,
               layer->weight_updates);
  }
  if (layer->bias_updates) {
    InitTensor(layer->output_size, 0, layer->bias_updates);
  }
  if (layer->bn_gamma_grads) {
    InitTensor(layer->output_size, 0, layer->bn_gamma_grads);
  }
  if (layer->bn_beta_grads) {
    InitTensor(layer->output_size, 0, layer->bn_beta_grads);
  }
  if (layer->output_normed) {
    InitTensor(layer->output_size * layer->batch_size, 0, layer->output_normed);
  }
  if (layer->output_before_norm) {
    InitTensor(layer->output_size * layer->batch_size, 0,
               layer->output_before_norm);
  }
}

void PrintWeight(Layer *layer) {
  int num_size = 8;
  int n = layer->input_size;
  int m = layer->output_size;
  int batch_size = layer->batch_size;

  printf("layer %s weight size = %d × %d + %d\n", layer->layer_name, n, m, m);
  printf("bias:\n");
  PrintGridOutline(m * (num_size + 3) - 1);
  printf("|");
  for (int i = 0; i < m; ++i) {
    float res = layer->biases[i];
    if (res >= 0) {
      printf(" %s |", FloatToString(num_size, res));
    } else {
      printf("%s |", FloatToString(num_size, res));
    }
  }
  printf("\n");
  PrintGridOutline(m * (num_size + 3) - 1);

  printf("weight :\n");
  PrintGridOutline(m * (num_size + 3) - 1);
  for (int i = 0; i < n; ++i) {
    printf("|");
    for (int j = 0; j < m; ++j) {
      float res = layer->weights[m * i + j];
      if (res >= 0) {
        printf(" %s |", FloatToString(num_size, res));
      } else {
        printf("%s |", FloatToString(num_size, res));
      }
    }
    printf("\n");
  }
  PrintGridOutline(m * (num_size + 3) - 1);
}

void PrintUpdate(Layer *layer) {
  int num_size = 8;
  int n = layer->input_size;
  int m = layer->output_size;
  int batch_size = layer->batch_size;
  printf("layer %s update size = %d × %d + %d\n", layer->layer_name, n, m, m);

  printf("bias update:\n");
  PrintGridOutline(m * (num_size + 3) - 1);
  printf("|");
  for (int i = 0; i < m; ++i) {
    float res = layer->bias_updates[i];
    if (res >= 0) {
      printf(" %s |", FloatToString(num_size, res));
    } else {
      printf("%s |", FloatToString(num_size, res));
    }
  }
  printf("\n");
  PrintGridOutline(m * (num_size + 3) - 1);

  printf("weight update:\n");
  PrintGridOutline(m * (num_size + 3) - 1);
  for (int i = 0; i < n; ++i) {
    printf("|");
    for (int j = 0; j < m; ++j) {
      float res = layer->weight_updates[m * i + j];
      if (res >= 0) {
        printf(" %s |", FloatToString(num_size, res));
      } else {
        printf("%s |", FloatToString(num_size, res));
      }
    }
    printf("\n");
  }
  PrintGridOutline(m * (num_size + 3) - 1);
}

void PrintInput(Layer *layer, int batch_num) {
  assert(layer->input);
  assert(layer->ground_truth);

  int num_size = 8;
  int n = layer->input_size;
  int batch_size = layer->batch_size;
  batch_num = batch_num > batch_size ? batch_size : batch_num;

  printf("layer %s input size = %d × %d:\n", layer->layer_name, batch_size, n);
  PrintGridOutline((n + 1) * (num_size + 3) - 1);

  for (int i = 0; i < batch_num; ++i) {
    printf("|");
    for (int j = 0; j < n + 1; ++j) {
      float res;
      if (j == n) {
        res = layer->ground_truth[i];
      } else {
        res = layer->input[n * i + j];
      }
      if (res >= 0) {
        printf(" %s |", FloatToString(num_size, res));
      } else {
        printf("%s |", FloatToString(num_size, res));
      }
    }
    printf("\n");
  }
  PrintGridOutline((n + 1) * (num_size + 3) - 1);
}

void PrintOutput(Layer *layer, int batch_num) {
  int num_size = 8;
  int n = layer->output_size;
  int batch_size = layer->batch_size;
  batch_num = batch_num > batch_size ? batch_size : batch_num;

  printf("layer: %s, output size = %d × %d:\n", layer->layer_name, batch_size,
         n);
  // PrintGridInnerline(n, 10);
  PrintGridOutline(n * (num_size + 3) - 1);
  // PrintGridColums(n , num_size + 2);
  // PrintGridOutline((n + 1) * (num_size + 3) - 1);
  for (int i = 0; i < batch_num; ++i) {
    // printf("| %d", i);
    // for (int j = 0; j < num_size - GetIntCharCount(i); ++j) {
    //   printf(" ");
    // }
    printf("|");
    for (int j = 0; j < n; ++j) {
      float res = layer->output[n * i + j];
      if (res >= 0) {
        printf(" %s |", FloatToString(num_size, res));
      } else {
        printf("%s |", FloatToString(num_size, res));
      }
    }
    printf("\n");
  }
  PrintGridOutline(n * (num_size + 3) - 1);
}

void PrintGrad(Layer *layer) {
  int num_size = 8;
  int n = layer->input_size;
  int m = layer->output_size;
  printf("layer %s weight size = %d × %d + %d:\n", layer->layer_name, n, m, m);
  printf("bias grad:\n");
  PrintGridOutline(m * (num_size + 3) - 1);
  printf("|");
  for (int i = 0; i < m; ++i) {
    float res = layer->bias_grads[i];
    if (res >= 0) {
      printf(" %s |", FloatToString(num_size, res));
    } else {
      printf("%s |", FloatToString(num_size, res));
    }
  }
  printf("\n");
  PrintGridOutline(m * (num_size + 3) - 1);

  printf("weight grad:\n");
  PrintGridOutline(m * (num_size + 3) - 1);
  for (int i = 0; i < n; ++i) {
    printf("|");
    for (int j = 0; j < m; ++j) {
      float res = layer->weight_grads[m * i + j];
      if (res >= 0) {
        printf(" %s |", FloatToString(num_size, res));
      } else {
        printf("%s |", FloatToString(num_size, res));
      }
    }
    printf("\n");
  }
  PrintGridOutline(m * (num_size + 3) - 1);
}

void PrintDelta(Layer *layer, int batch_num) {
  int num_size = 8;
  int n = layer->output_size;
  int batch_size = layer->batch_size;
  batch_num = batch_num > batch_size ? batch_size : batch_num;
  printf("layer %s delta size = %d × %d:\n", layer->layer_name, batch_size, n);
  PrintGridOutline(n * (num_size + 3) - 1);
  for (int i = 0; i < batch_num; ++i) {
    printf("|");
    for (int j = 0; j < n; ++j) {
      float res = layer->delta[n * i + j];
      if (res >= 0) {
        printf(" %s |", FloatToString(num_size, res));
      } else {
        printf("%s |", FloatToString(num_size, res));
      }
    }
    printf("\n");
  }
  PrintGridOutline(n * (num_size + 3) - 1);
}
