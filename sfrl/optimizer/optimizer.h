#ifndef OPTIMIZER_H
#define OPTIMIZER_H

/**
 * 优化方法
 * 这里讲的非常详细 https://d2l.ai/chapter_optimization/sgd.html
 **/
typedef enum { ADAM, SGD, ADAGRAD, RMSPROP } OptType;

void SgdOptimizer(int input_size, int output_size, float *weights,
                  float *weight_grads, float *biases, float *bias_grads, float *grad_cum_w,
                  float *grad_cum_b, float lr, float momentum);
void AdaGradOptimizer(int input_size, int output_size, float *weights,
                      float *weight_grads, float *biases, float *bias_grads,
                      float *grad_cum_square_w, float *grad_cum_square_b, float lr);
void RmsPropOptimizer(int input_size, int output_size, float *weights,
                      float *weight_grads, float *biases, float *bias_grads,
                      float *grad_cum_square_w, float *grad_cum_square_b, float lr, float decay);
void AdamOptimizer(int input_size, int output_size, float *weights,
                   float *weight_grads, float *biases, float *bias_grads, float *grad_cum_w,
                   float *grad_cum_square_w, float *grad_cum_b, float *grad_cum_square_b, float lr,
                   float beta_1, float beta_2);
#endif