#ifndef LOSS_LAYER_H
#define LOSS_LAYER_H

#include "sfrl/activations/activations.h"
#include "sfrl/layer/layer.h"
#include "sfrl/network/network.h"
#include "sfrl/utils/init.h"

typedef Layer DenseLayer;

DenseLayer MakeDenseLayer(int batch_size, int input_size, int output_size,
                          ActiType acti_type, InitType init_type);
void ForwardDenseLayer(DenseLayer layer, NetWork net);
void BackwardDenseLayer(DenseLayer layer, NetWork net);
void UpdateDenseLayer(DenseLayer layer, int batch_size, float learning_rate,
                      float momentum, float decay);

#endif