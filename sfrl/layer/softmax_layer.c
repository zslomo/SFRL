#include "softmax_layer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../loss/loss.h"
#include "../utils/blas.h"

SoftmaxLayer *MakeSoftmaxLayer(int batch_size, int input_size, int pre_layer_cnt, int post_layer_cnt,
                               char *layer_name) {
  SoftmaxLayer *layer = calloc(1, sizeof(SoftmaxLayer));
  layer->layer_type = SOFTMAX;
  layer->batch_size = batch_size;
  layer->input_size = input_size;  // softmax_layer的输入输出元素相同 其实就是类别个数
  layer->output_size = input_size;
  layer->layer_name = layer_name;

  assert(pre_layer_cnt <= 1);
  layer->pre_layer_cnt = pre_layer_cnt;
  if (pre_layer_cnt > 0) {
    layer->post_layers = calloc(pre_layer_cnt, sizeof(Layer *));
  }
  layer->post_layer_cnt = post_layer_cnt;
  if (post_layer_cnt > 0) {
    layer->post_layers = calloc(post_layer_cnt, sizeof(Layer *));
  }

  layer->input = calloc(input_size * batch_size, sizeof(float));
  layer->output = calloc(input_size * batch_size, sizeof(float));
  layer->delta = calloc(input_size * batch_size, sizeof(float));
  layer->temperature = 1.0;
  layer->forward = ForwardSoftmaxLayer;
  layer->backward = BackwardSoftmaxLayer;
  layer->print_input = PrintInput;
  layer->print_output = PrintOutput;
  layer->print_delta = PrintDelta;
  layer->reset = ResetLayer;

  return layer;
}

void ForwardSoftmaxLayer(SoftmaxLayer *layer, Network *net) {
  if (layer->pre_layers) {
    assert(layer->pre_layers[0]->output_size == layer->input_size);
  }
  memcpy(layer->input, net->input, layer->input_size * layer->batch_size * sizeof(float));
  SoftmaxBatch(net->input, layer->input_size, layer->batch_size, layer->temperature, layer->output);
  memcpy(net->pred, layer->output, layer->batch_size * layer->output_size * sizeof(float));
}

/**
 * softmax 的反向传播函数
 * 这里就简单多了，CE的loss就是SoftMax + CE形式下的，相当于在CE里已经算好啦
 * 所以这里的倒数就是上一步的，
 * */
void BackwardSoftmaxLayer(SoftmaxLayer *layer, Network *net) {
  // 注意，这里的net->delta是 i+1层的 delta也就是 反向传播的上一层
  // 计算后赋值给当前层的delta layer->delta
  memcpy(net->delta, layer->delta, layer->batch_size * layer->input_size * sizeof(float));
  // AxpyTensor(layer->batch_size * layer->input_size, 1, layer->delta,
  // net->delta);
}

/**
 *  batch softmax函数，分batch 计算softmax，由softmax layer调用
 *  batch_size 指的是每个batch的大小，
 *  n 是分类个数
 * */
void SoftmaxBatch(float *input, int n, int batch_size, float temp, float *output) {
  for (int i = 0; i < batch_size; ++i) {
    int offset = i * n;
    // printf("input offset: %d, value: %f, %f\n", offset, (input + offset)[0],
    // (input + offset)[1]);
    SoftmaxCore(input + offset, n, temp, output + offset);
  }
}

/**
 *  softmax函数
 *  temp 温度，详见
 * https://stackoverflow.com/questions/58764619/why-should-we-use-temperature-in-softmax
 * */
void SoftmaxCore(float *input, int n, float temp, float *output) {
  assert(input);
  assert(n > 0);
  float sum = 0;
  // 这个largest 为了计算稳定，防止溢出具体见
  // http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
  float largest = -FLT_MAX;
  for (int i = 0; i < n; ++i) {
    if (input[i] > largest) largest = input[i];
  }
  for (int i = 0; i < n; ++i) {
    float e = exp(input[i] / temp - largest / temp);
    sum += e;
    output[i] = e;
  }
  // 归一化转换为概率
  for (int i = 0; i < n; ++i) {
    output[i] /= sum;
  }
}
