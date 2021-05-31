#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "softmax_layer.h"
#include "../../sfrl/loss/loss.h"

SoftmaxLayer MakeSoftmaxLayer(int batch_size, int input_size) {
  SoftmaxLayer layer = {0};
  layer.layer_type = SOFTMAX;
  layer.batch_size = batch_size;
  layer.input_size = input_size; // softmax_layer的输入输出元素相同
  layer.output_size = input_size;

  layer.output = calloc(input_size * batch_size, sizeof(float));
  layer.delta = calloc(input_size * batch_size, sizeof(float));

  layer.forward = ForwardSoftmaxLayer;
  layer.backward = BackwardSoftmaxLayer;

  return layer;
}

void ForwardSoftmaxLayer(const SoftmaxLayer *layer, NetWork *net) {
  SoftmaxBatch(net->input, layer->input_size, layer->batch_size, layer->input_size,
               layer->temperature, layer->output);
}

void BackwardSoftmaxLayer(const SoftmaxLayer *layer, NetWork *net) {
  // 注意，这里的net->delta是 i+1层的 delta也就是 反向传播的上一层
  // 计算后赋值给当前层的delta layer->delta
  BackwardSoftmax(layer->output, layer->delta, layer->input_size, layer->batch_size,
                  layer->input_size, layer->temperature, net->delta);
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
    if (input[i] > largest)
      largest = input[i];
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

/**
 *  batch softmax函数，分batch 计算softmax，由softmax layer调用
 *  batch_size 指的是每个batch的大小，
 *  batch_offset
 * 指的是输入的每个tensor的大小，因为都被flat到一维数组，需要该参数确定下一个tensor的位置
 * */
void SoftmaxBatch(float *input, int n, int batch_size, int batch_offset, float temp,
                  float *output) {
  for (int i = 0; i < batch_size; ++i) {
    int offset = i * batch_offset;
    SoftmaxCore(input + offset, n, temp, output + offset);
  }
}

/**
 * softmax 的反向传播函数
 * */
void BackwardSoftmaxCore(float *output, float *delta_output, int n, float temp,
                         float *delta_input) {
  float dot = dotTensor(n, output, delta_output);
  float temp_inv = 1.0 / temp;
  for (int i = 0; i < n; ++i) {
    delta_input[i] += temp_inv * output[i] * (delta_output[i] - dot);
  }
}

void BackwardSoftmax(float *output, float *delta_output, int n, int batch_size, int batch_offset,
                     float temp, float *delta_input) {
  int g, b;
  int offset;
  for (int i = 0; i < batch_size; ++i) {
    int offset = i * batch_offset;
    BackwardSoftmaxCore(output + offset, delta_output + offset, n, temp, delta_input + offset);
  }
}
