
#include "sfrl/loss/loss.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include "sfrl/utils/blas.h"

// MSE 这里不求和
void MeanSquareError(int n, float *pred, float *truth, float *delta,
                     float *error) {
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    error[i] = diff * diff / n;
    // delta的 每一项可能非常小，/n 会弥散，
    // 这里用
    delta[i] = diff;
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

/**
 *  batch softmax函数，分batch 计算softmax，由softmax layer调用
 *  batch_size 指的是每个batch的大小，
 *  batch_offset
 * 指的是输入的每个tensor的大小，因为都被flat到一维数组，需要该参数确定下一个tensor的位置
 * */
void SoftmaxBatch(float *input, int n, int batch_size, int batch_offset,
                  float temp, float *output) {
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
  float dot = dotd1(n, output, 1, delta_output, 1);
  float temp_inv = 1.0 / temp;
  for (int i = 0; i < n; ++i) {
    delta_input[i] += temp_inv * output[i] * (delta_output[i] - dot);
  }
}

void BackwardSoftmax(float *output, float *delta_output, int n, int batch_size,
                     int batch_offset, float temp, float *delta_input) {
  int g, b;
  int offset;
  for (int i = 0; i < batch_size; ++i) {
    int offset = i * batch_offset;
    BackwardSoftmaxCore(output + offset, delta_output + offset, n, temp,
                        delta_input + offset);
  }
}

void CrossEntropy(int n, float *pred, float *truth, float *delta) {
  assert(pred);
  assert(truth);
  for (int i = 0; i < n; ++i) {
    delta[i] = truth[i] * log(pred[i]);
  }
}

void SoftMaxWithCrossEntropy(float *input, float *truth, int n, float temp,
                             float *output) {
  float *pred = calloc(n, sizeof(float));
  SoftmaxCore(input, n, temp, pred);
  CrossEntropy(n, pred, truth, output);
}

// 加权 交叉熵，policy gradient 会使用
void CrossEntropyWithWeight(int n, float *pred, float *weight, float *delta) {
  assert(pred);
  assert(weight);
  for (int i = 0; i < n; ++i) {
    delta[i] = log(pred[i]) * weight[i];
  }
}