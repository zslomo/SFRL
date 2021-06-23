#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H
#include "../activation/activation.h"
#include "../network/network.h"
#include "../utils/init.h"
#include "base_layer.h"

typedef Layer DenseLayer;

DenseLayer *MakeDenseLayer(int batch_size, int input_size, int output_size, int pre_layer_cnt,
                           int post_layer_cnt, ActiType acti_type, InitType init_type, int seed,
                           char *layer_name);
void ForwardDenseLayer(DenseLayer *layer, Network *net);
void BackwardDenseLayer(DenseLayer *layer, Network *net);
void UpdateDenseLayer(DenseLayer *layer, Network *net);

#endif