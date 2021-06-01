#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "base_layer.h"
#include "dense_layer.h"
#include "../../sfrl/activation/activation.h"
#include "../../sfrl/optimizer/optimizer.h"
#include "../../sfrl/utils/blas.h"

DenseLayer MakeDenseLayer(int batch_size, int input_size, int output_size, ActiType acti_type,
                          InitType init_type) {
  DenseLayer layer = {0};
  layer.layer_type = DENSE;
  layer.batch_size = batch_size;
  layer.input_size = input_size;
  layer.output_size = output_size;
  layer.acti_type = acti_type;

  layer.output = calloc(input_size * batch_size, sizeof(float));
  // w, b, delta
  layer.delta = calloc(input_size * batch_size, sizeof(float));
  layer.weights = calloc(output_size * input_size, sizeof(float));
  layer.biases = calloc(output_size, sizeof(float));
  layer.weight_grads = calloc(input_size * output_size, sizeof(float));
  layer.bias_grads = calloc(output_size, sizeof(float));
  InitLayer(layer.weights, layer.biases, input_size, output_size, init_type);

  layer.forward = ForwardDenseLayer;
  layer.backward = BackwardDenseLayer;
  layer.update = UpdateDenseLayer;

  return layer;
}

void UpdateDenseLayer(DenseLayer *layer, NetWork *net) { UpdateLayer(layer, net); }

void ForwardDenseLayer(DenseLayer *layer, NetWork *net) {
  // 最终输出的是一个flat后的一维tensor 大小是output_size * batch_size

  int output_tensor_size = layer->output_size * layer->batch_size;
  FillTensorBySingleValue(output_tensor_size, layer->output, 0);

  /**
   *  计算 intput × weights->T
   *  维度是 M*K × K*N = M*N
   *  A input M*K
   *  B weights N*K
   *  C output M*N
   *  M batch_size A的行
   *  N output_size B->T的列，就是B的行，ldc是 N
   *  K input_size, A的列，所以 lda ldb也是 K
   *  ALPHA 和 BETA这里都是1
   **/
  int TransA = 0;
  int TransB = 1;
  Gemm(TransA, TransB, layer->batch_size, layer->output_size, layer->input_size, 1, 1, net->input,
       layer->input_size, layer->weights, layer->input_size, layer->output, layer->output_size);

  /**
    计算 intput × weights->T + bias
   **/
  for (int i = 0; i < layer->batch_size; ++i) {
    AxpyTensor(layer->output_size, 1, layer->biases, layer->output + i * layer->input_size);
  }

  /**
   *  计算 f(intput × weights->T + bias)
   **/
  ActivateTensor(layer->output, output_tensor_size, layer->acti_type);
}

void BackwardDenseLayer(DenseLayer *layer, NetWork *net) {
  int output_tensor_size = layer->output_size * layer->batch_size;
  /**
   *  计算 delta
   *  delta = f'(x) * delta_tmp
   *  ndelta_tmp 是前一层计算好的 delta_tmp = delta(i+1) × weights
   **/
  GradientTensor(layer->output, output_tensor_size, layer->acti_type, layer->delta);

  /**
   *  计算 bias_grads = delta
   **/
  for (int i = 0; i < layer->batch_size; ++i) {
    AxpyTensor(layer->output_size, 1, layer->delta + i * layer->output_size, layer->bias_grads);
  }

  /**
   *  计算 当前层的weight_grads = delta->T × input
   *  维度是 M*K × K*N = M*N
   *  A delta N*M
   *  B input K*N
   *  C weight_grads M*N
   *  M lda output_size A的列 A->T的行
   *  N ldb ldc input_size, B的列
   *  K batch_size, A的行 A->T的列
   *  ALPHA 和 BETA这里都是1
   **/
  int TransA = 1;
  int TransB = 0;
  Gemm(TransA, TransB, layer->output_size, layer->input_size, layer->batch_size, 1, 1, layer->delta,
       layer->output_size, net->input, layer->input_size, layer->weight_grads, layer->input_size);

  /**
   *  计算 后一层的delta，即net->delta
   *  反向传播的delta要在前一层计算好，这样的话当前层的权重梯度(也就是weight_grads)
   *  就可以直接用(f'(x) * delta)->T × input算出来了 注意最后一层的时候是没有后一层的，此时
   *net->delta
   *== null 不需要计算 net->delta = delta_tmp = delta × weights 维度是 M*K × K*N = M*N A delta M*K B
   *weights K*N C net->delta M*N M batch_size A的行 N ldb ldc input_size, B的列 K lda output_size,
   *A的列 B的行 ALPHA 和 BETA这里都是1
   **/
  if (net->delta) {
    int TransA = 0;
    int TransB = 0;
    Gemm(TransA, TransB, layer->batch_size, layer->input_size, layer->output_size, 1, 1,
         layer->delta, layer->output_size, layer->weights, layer->input_size, net->delta,
         layer->input_size);
  }
}
