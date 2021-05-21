#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "sfrl/activations/activations.h"
#include "sfrl/layer/layer.h"
#include "sfrl/network/network.h"

typedef Layer ActivationLayer;
Layer MakeActivationLayer(int batch_size, int input_size, ActiType acti_type);
void ForwardActivationLayer(ActivationLayer layer, NetWork net);
void BackwardActivationLayer(ActivationLayer layer, NetWork net);

void ActivateTensor(float *TensorX, const int size, const ActiType acti_type);
void GradientTensor(const float *TensorX, const int size, const ActiType acti_type, float *delta);

#endif

