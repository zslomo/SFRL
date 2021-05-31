#ifndef INIT_H
#define INIT_H

/**
 * 初始化类型
 **/
typedef enum { NORMAL, UNIFORM } InitType;

float rand_uniform(float min, float max);
float rand_normal();
void InitLayer(float *weights, float *biases, int input_size, int output_size, InitType init_type);

#endif