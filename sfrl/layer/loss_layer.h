#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H
#include "base_layer.h"
#include "../network/network.h"

typedef Layer LossLayer;

void ForwardLossLayer(LossLayer *loss_layer, NetWork *net);
void BackwardLossLayer(LossLayer *loss_layer, NetWork *net);
LossLayer MakeLossLayer(int batch_size, int input_size, LossType loss_type);

#endif