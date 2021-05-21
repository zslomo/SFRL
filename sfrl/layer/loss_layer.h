#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H
#include "sfrl/layer/layer.h"
#include "sfrl/network/network.h"

/**
 * 损失函数
**/
typedef enum {
  MSE,
  SOFTMAX,
  CEW
} LossType;

typedef Layer LossLayer;

void ForwardLossLayer(LossLayer loss_layer, NetWork net);
void BackwardLossLayer(LossLayer loss_layer, NetWork net);

#endif