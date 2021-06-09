#include "dense_layer.h"
#include "../activation/activation.h"
#include "../optimizer/optimizer.h"
#include "../utils/blas.h"
#include "../utils/utils.h"
#include "base_layer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DenseLayer MakeDenseLayer(int batch_size, int input_size, int output_size, ActiType acti_type,
                          InitType init_type, char *layer_name) {
  DenseLayer layer = {0};
  layer.layer_type = DENSE;
  layer.layer_name = layer_name;
  layer.batch_size = batch_size;
  layer.input_size = input_size;
  layer.output_size = output_size;
  layer.acti_type = acti_type;

  layer.output = calloc(output_size * batch_size, sizeof(float));
  layer.input = calloc(input_size * batch_size, sizeof(float));
  // w, b, delta
  layer.delta = calloc(batch_size * output_size, sizeof(float));
  layer.weights = calloc(input_size * output_size, sizeof(float));
  layer.biases = calloc(output_size, sizeof(float));
  layer.weight_grads = calloc(input_size * output_size, sizeof(float));
  layer.bias_grads = calloc(output_size, sizeof(float));
  layer.weight_updates = calloc(input_size * output_size, sizeof(float));
  layer.bias_updates = calloc(output_size, sizeof(float));
  InitLayer(layer.weights, layer.biases, input_size, output_size, init_type);

  layer.forward = ForwardDenseLayer;
  layer.backward = BackwardDenseLayer;
  layer.update = UpdateDenseLayer;
  layer.print_input = PrintInput;
  layer.print_output = PrintOutput;
  layer.print_weight = PrintWeight;
  layer.print_grad = PrintGrad;
  layer.print_delta = PrintDelta;
  layer.print_update = PrintUpdate;
  layer.reset = ResetLayer;

  return layer;
}

void UpdateDenseLayer(DenseLayer *layer, Network *net) { UpdateLayer(layer, net); }

void ForwardDenseLayer(DenseLayer *layer, Network *net) {
  CopyTensor(layer->input_size * net->batch_size, net->input, layer->input);
  int output_tensor_size = layer->output_size * net->batch_size;
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
  Gemm(TransA, TransB, net->batch_size, layer->output_size, layer->input_size, 1, 1, net->input,
       layer->input_size, layer->weights, layer->output_size, layer->output, layer->output_size);

  /**
    计算 + bias
   **/
  for (int i = 0; i < net->batch_size; ++i) {
    AxpyTensor(layer->output_size, 1, layer->biases, layer->output + i * layer->output_size);
  }
  /**
   *  计算 f(intput × weights + bias)
   **/
  ActivateTensor(layer->output, output_tensor_size, layer->acti_type);
}

void BackwardDenseLayer(DenseLayer *layer, Network *net) {
  int output_tensor_size = layer->output_size * net->batch_size;
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
  for (int i = 0; i < net->batch_size; ++i) {
    CopyTensor(layer->output_size, layer->delta + i * layer->output_size, layer->bias_grads);
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
   *  lda  batch_size
   *  ldb  output_size
   *  ldc  output_size
   **/
  int TransA = 1;
  int TransB = 0;
  Gemm(TransA, TransB, layer->input_size, layer->output_size, net->batch_size, 1, 1, layer->input,
       net->batch_size, layer->delta, layer->output_size, layer->weight_grads, layer->output_size);

  /**
   *  计算 后一层的delta，即net->delta
   *  反向传播的delta要在前一层计算好，这样的话当前层的权重梯度(也就是weight_grads)
   *  就可以直接用input.T × (f'(x) * delta)算出来了
   *  A delta batch_size * output_size
   *  B weights input_size * output_size
   *  B.T       output_size * input_size
   *  C net->delta batch_size * input_size
   *  M A 行    batch_size
   *  N B.T 列  input_size
   *  K A 列    output_size
   *  lda batch_size
   *  ldb input_size
   *  ldc input_size
   **/

  if (net->delta) {
    int TransA = 0;
    int TransB = 1;
    Gemm(TransA, TransB, net->batch_size, layer->input_size, layer->output_size, 1, 1, layer->delta,
         layer->output_size, layer->weights, layer->input_size, net->delta, layer->input_size);
  }
}
