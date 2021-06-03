#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H
#include "base_layer.h"
#include "../activation/activation.h"
#include "../network/network.h"
#include "../utils/init.h"

typedef Layer DenseLayer;

DenseLayer MakeDenseLayer(int batch_size, int input_size, int output_size, ActiType acti_type,
                          InitType init_type);
void ForwardDenseLayer(DenseLayer *layer, NetWork *net);
void BackwardDenseLayer(DenseLayer *layer, NetWork *net);
void UpdateDenseLayer(DenseLayer *layer, NetWork *net);

#endif