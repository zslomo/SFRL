#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include "init.h"

#define TWO_PI 6.283185307

void InitLayer(float *weights, float *biases, int input_size, int output_size,
               InitType init_type) {
  float scale = sqrt(2. / input_size);
  if (init_type == UNIFORM) {

    for (int i = 0; i < output_size * input_size; ++i) {
      weights[i] = scale * rand_uniform(-1, 1);
    }
  } else {
    for (int i = 0; i < output_size * input_size; ++i) {
      weights[i] = scale * rand_normal();
    }
  }

  // 初始化所有偏置值为0
  for (int i = 0; i < output_size; ++i) {
    biases[i] = 0;
  }
}

// 返回均匀分布随机数
float rand_uniform(float min, float max) {
  if (max < min) {
    float swap = min;
    min = max;
    max = swap;
  }
  return ((float)rand() / RAND_MAX * (max - min)) + min;
}

// Box-Muller算法 返回标准正态分布随机数（float）
float rand_normal() {
  static int haveSpare = 0;
  static double rand1, rand2;

  if (haveSpare) {
    haveSpare = 0;
    return sqrt(rand1) * sin(rand2);
  }

  haveSpare = 1;

  rand1 = rand() / ((double)RAND_MAX);
  if (rand1 < 1e-100) {
    rand1 = 1e-100;
  }

  rand1 = -2 * log(rand1);

  rand2 = (rand() / ((double)RAND_MAX)) * TWO_PI;

  return sqrt(rand1) * cos(rand2);
}