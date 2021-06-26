#include "dense_layer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../activation/activation.h"
#include "../optimizer/optimizer.h"
#include "../utils/blas.h"
#include "../utils/utils.h"
#include "base_layer.h"

DenseLayer *MakeDenseLayer(int batch_size, int input_size, int output_size, int pre_layer_cnt, int post_layer_cnt,
                           ActiType acti_type, InitType init_type, int seed, char *layer_name) {
  DenseLayer *layer = calloc(1, sizeof(DenseLayer));
  layer->layer_type = DENSE;
  layer->layer_name = layer_name;
  layer->batch_size = batch_size;
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->acti_type = acti_type;
  assert(pre_layer_cnt <= 1);
  layer->pre_layer_cnt = pre_layer_cnt;
  if (pre_layer_cnt > 0) {
    layer->post_layers = calloc(pre_layer_cnt, sizeof(Layer *));
  }
  layer->post_layer_cnt = post_layer_cnt;
  if (post_layer_cnt > 0) {
    layer->post_layers = calloc(post_layer_cnt, sizeof(Layer *));
  }

  layer->output = calloc(output_size * batch_size, sizeof(float));
  layer->input = calloc(input_size * batch_size, sizeof(float));
  // w, b, delta
  layer->delta = calloc(batch_size * output_size, sizeof(float));
  layer->weights = calloc(input_size * output_size, sizeof(float));
  layer->biases = calloc(output_size, sizeof(float));
  layer->weight_grads = calloc(input_size * output_size, sizeof(float));
  layer->bias_grads = calloc(output_size, sizeof(float));
  layer->weight_updates = calloc(input_size * output_size, sizeof(float));
  layer->bias_updates = calloc(output_size, sizeof(float));
  InitLayer(layer->weights, layer->biases, input_size, output_size, init_type, seed);

  layer->forward = ForwardDenseLayer;
  layer->backward = BackwardDenseLayer;
  layer->update = UpdateDenseLayer;
  layer->print_input = PrintInput;
  layer->print_output = PrintOutput;
  layer->print_weight = PrintWeight;
  layer->print_grad = PrintGrad;
  layer->print_delta = PrintDelta;
  layer->print_update = PrintUpdate;
  layer->reset = ResetLayer;

  return layer;
}

void UpdateDenseLayer(DenseLayer *layer, Network *net) { UpdateLayer(layer, net); }

void ForwardDenseLayer(DenseLayer *layer, Network *net) {
  if (layer->pre_layers) {
    assert(layer->pre_layers[0]->output_size == layer->input_size);
  }
  memcpy(layer->input, net->input, layer->input_size * layer->batch_size * sizeof(float));
  int output_tensor_size = layer->output_size * layer->batch_size;
  FillTensorBySingleValue(output_tensor_size, layer->output, 0);
  /**
   *  计算 intput × weights
   *  A input batch_size * input_size
   *  B weights input_size * output_size
   *  C output batch_size * output_size
   *  M A 行  batch_size
   *  N B 列  output_size
   *  K A 列  input_size
   *  lda input_size
   *  ldb output_size
   *  ldc output_size
   **/
  int TransA = 0;
  int TransB = 0;
  Gemm(TransA, TransB, layer->batch_size, layer->output_size, layer->input_size, 1, 1, net->input, layer->input_size,
       layer->weights, layer->output_size, layer->output, layer->output_size);

  /**
    计算 + bias
   **/
  for (int i = 0; i < layer->batch_size; ++i) {
    AxpyTensor(layer->output_size, 1, layer->biases, layer->output + i * layer->output_size);
  }
  /**
   *  计算 f(intput × weights + bias)
   **/
  ActivateTensor(layer->output, output_tensor_size, layer->acti_type);
}

void BackwardDenseLayer(DenseLayer *layer, Network *net) {
  int output_tensor_size = layer->output_size * layer->batch_size;
  /**
   *  计算 delta
   *  delta = f'(x) * delta_pre, delta_pre在之前就算好了。
   *  就是net->delta，上一步会指向这步的layer-delta
   *  在这里直接乘 激活函数的导数
   **/
  GradientTensor(layer->output, output_tensor_size, layer->acti_type, layer->delta);
  /**
   *  计算 bias_grads = delta
   **/
  for (int i = 0; i < layer->batch_size; ++i) {
    memcpy(layer->bias_grads, layer->delta + i * layer->output_size, layer->output_size * sizeof(float));
    // printf("bias_grads[0] = %f\n", layer->bias_grads[0]);
  }

  /**
   *  计算 当前层的weight_grads = input.T × delta
   *  A.T     input_size * batch_size
   *  B delta batch_size * output_size
   *  C grad  input_size * output_size
   *  M A.T 行  input_size
   *  N B 列    output_size
   *  K A 列    batch_size
   *  lda  input_size
   *  ldb  output_size
   *  ldc  output_size
   **/
  int TransA = 1;
  int TransB = 0;

  Gemm(TransA, TransB, layer->input_size, layer->output_size, layer->batch_size, 1, 1, layer->input, layer->input_size,
       layer->delta, layer->output_size, layer->weight_grads, layer->output_size);
  /**
   *  计算 后一层的delta，即net->delta
   *  A delta batch_size * output_size
   *  B weights input_size * output_size
   *  B.T       output_size * input_size
   *  C net->delta batch_size * input_size
   *  M A 行    batch_size
   *  N B.T 列  input_size
   *  K A 列    output_size
   *  lda output_size
   *  ldb output_size
   *  ldc input_size
   **/

  if (net->delta) {
    int TransA = 0;
    int TransB = 1;
    Gemm(TransA, TransB, layer->batch_size, layer->input_size, layer->output_size, 1, 1, layer->delta,
         layer->output_size, layer->weights, layer->output_size, net->delta, layer->input_size);
  }
}
