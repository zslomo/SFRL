#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H
#include "base_layer.h"
#include "../../sfrl/network/network.h"

typedef Layer LossLayer;

void ForwardLossLayer(LossLayer *loss_layer, NetWork *net);
void BackwardLossLayer(LossLayer *loss_layer, NetWork *net);

#endif