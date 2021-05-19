#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "sfrl/activations/activations.h"
#include "sfrl/layer/layer.h"
#include "sfrl/network/network.h"

Layer MakeActivationLayer(int batch, int inputs, ActiType acti_type);

void ForwardActivationLayer(Layer layer, NetWork net);
void BackwardActivationLayer(Layer layer, NetWork net);

#endif

