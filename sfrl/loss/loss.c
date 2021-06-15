
#include "loss.h"
#include "../utils/blas.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// MSE 这里不求和
void MeanSquareError(int n, float *pred, float *truth, float *loss) {
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    loss[i] = diff * diff / n;
  }
}

void BackwardMeanSquareError(int n, float *pred, float *truth, float *delta) {
  assert(n > 0);
  for (int i = 0; i < n; ++i) {
    float diff = truth[i] - pred[i];
    // delta的 每一项可能非常小，/n 会弥散，
    delta[i] = diff;
  }
}

void CrossEntropy(int batch_size, int class_num, float *pred, float *truth, float *loss,
                  int weight_ce) {
  assert(pred);
  assert(truth);
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < class_num; ++j) {
      // weight_ce 是强化学习的特殊形式，强化学习没有监督信号，
      // truth并不是一个类别标签 而是权重
      int t = weight_ce ? truth[i] : ((int)truth[i] ^ j) ^ 1;
      loss[class_num * i + j] = -t * log(pred[class_num * i + j]);
      // printf("i: %d, j: %d, loss: %f, loss[i]: %f, t: %d, log: %f, pred : %f, turh: %d, xor: %d \n",
      //        i, j, loss[class_num * i + j], t * log(pred[class_num * i + j]), t, log(pred[class_num * i + j]),
      //        pred[class_num * i + j], (int)truth[i], ((int)truth[i] ^ j) ^ 1);
    }
  }
}

void BackwardCrossEntropy(int batch_size, int class_num, float *pred, float *truth, float *delta,
                          int weight_ce) {
  assert(pred);
  assert(truth);
  float eps = 1e-8;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < class_num; ++j) {
      int t = weight_ce ? truth[i] : ((int)truth[i] ^ j) ^ 1;
      delta[class_num * i + j] = t - pred[class_num * i + j];
      // printf("i:%d, j:%d, delta:%f, t: %d, pred: %f\n", i, j, delta[class_num * i + j], t,
      //        pred[class_num * i + j]);
    }
  }
}

char *GetLossStr(LossType loss_type) {
  char *loss_str;
  if (loss_type == MSE) {
    loss_str = "MeanSquareError";
  } else if (loss_type == CE) {
    loss_str = "CrossEntropy";
  } else if (loss_type == CEW) {
    loss_str = "CrossEntropyWeight";
  } else {
    loss_str = "loss";
  }
  return loss_str;
}