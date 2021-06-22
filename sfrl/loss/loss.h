#ifndef LOSS_H
#define LOSS_H
#include "../type/type.h"

/**
 * todo：正则项
 **/
void MeanSquareError(int n, float *pred, float *truth, float *loss);
void BackwardMeanSquareError(int n, float *pred, float *truth, float *delta);
void CrossEntropy(int batch_size, int class_num, float *pred, float *truth, float *loss,
                  int weight_ce);
void BackwardCrossEntropy(int batch_size, int class_num, float *pred, float *truth, float *delta,
                          int weight_ce);
char *GetLossStr(LossType loss_type);

#endif