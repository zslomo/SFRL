#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H
#include "base_layer.h"
#include "../network/network.h"

typedef Layer LossLayer;

void ForwardLossLayer(LossLayer *loss_layer, Network *net);
void BackwardLossLayer(LossLayer *loss_layer, Network *net);
LossLayer *MakeLossLayer(int batch_size, int input_size, int output_size, LossType loss_type, char *layer_name);

#endif