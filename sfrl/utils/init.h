#ifndef INIT_H
#define INIT_H

/**
 * 初始化类型
 **/
typedef enum { NORMAL, UNIFORM, DEBUG } InitType;

float rand_uniform(float min, float max, int seed);
float rand_normal(int seed);
void InitLayer(float *weights, float *biases, int input_size, int output_size, InitType init_type, int seed);

#endif