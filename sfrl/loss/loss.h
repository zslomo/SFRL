#ifndef LOSS_H
#define LOSS_H

/**
 * 损失函数
**/
typedef enum {
  MSE,
  CE,
  CEW
} LossType;

/**
 * todo：正则项
 **/
void MeanSquareError(int n, float *pred, float *truth, float *error);
void BackwardMeanSquareError(int n, float *pred, float *truth, float *delta);
void CrossEntropy(int n, float *pred, float *truth, float *error);
void BackwardCrossEntropy(int n, float *pred, float *truth, float *delta);
void CrossEntropyWithWeight(int n, float *pred, float *weight, float *error);
void BackwardCrossEntropyWithWeight(int n, float *pred, float *weight, float *delta);

#endif