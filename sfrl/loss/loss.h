#ifndef LOSS_H
#define LOSS_H
/**
 * todo：正则项
 **/
void Mse(int n, float *pred, float *truth, float *delta, float *error);
void SoftMax(float *input, int n, float temp, float *output);
void CrossEntropy(int n, float *pred, float *truth, float *delta);
void SoftMaxWithCrossEntropy(float *input, float *truth, int n, float temp,
                             float *output);
void CrossEntropyWithWeight(int n, float *pred, float *weight, float *delta);

#endif