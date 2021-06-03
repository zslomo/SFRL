#ifndef LOSS_H
#define LOSS_H

/**
 * 损失函数
 **/
typedef enum { MSE, CE, CEW } LossType;

/**
 * todo：正则项
 **/
void MeanSquareError(int n, float *pred, float *truth, float *error);
void BackwardMeanSquareError(int n, float *pred, float *truth, float *delta);
void CrossEntropy(int batch_size, int class_num, float *pred, float *truth, float *error,
                  int weight_ce);
void BackwardCrossEntropy(int batch_size, int class_num, float *pred, float *truth, float *delta,
                          int weight_ce);

#endif