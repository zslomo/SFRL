#ifndef OPTIMIZER_H
#define OPTIMIZER_H

/**
 * 优化方法
 * 这里讲的非常详细 https://d2l.ai/chapter_optimization/sgd.html
 **/
typedef enum { ADAM, SGD, RMSPROP } OptType;

void SgdOptimizer(int input_size, int output_size, int batch_size, float *weights,
                  float *weight_updates, float *biases, float *bias_updates, float momentum,
                  float lr, float decay);
void SgdOptimizer(int size, float *weights, float *gradient, float momentum, float lr, float decay);
#endif