#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H

#include "sfrl/layer/layer.h"
#include "sfrl/network/network.h"

typedef Layer SoftmaxLayer;

SoftmaxLayer MakeSoftmaxLayer(int batch_size, int input_size);
void ForwardSoftmaxLayer(const SoftmaxLayer layer, NetWork net);
void BackwardSoftmaxLayer(const SoftmaxLayer layer, NetWork net);

#endif