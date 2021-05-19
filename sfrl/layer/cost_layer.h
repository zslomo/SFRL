#ifndef COST_LAYER_H
#define COST_LAYER_H
#include "sfrl/layer/layer.h"
#include "sfrl/network/network.h"

/**
 * 损失函数
**/
typedef enum {
  DENSE,
  NORMALIZATION,
  BATCHNORMALIZATION,
  DROPOUT,
  ACTIVE,
  SOFTMAX,
  COST
} CostType;

typedef Layer CostLayer;

#endif