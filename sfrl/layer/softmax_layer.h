#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "base_layer.h"
#include "../network/network.h"

typedef Layer SoftmaxLayer;

SoftmaxLayer MakeSoftmaxLayer(int batch_size, int input_size, char *layer_name);
void ForwardSoftmaxLayer(SoftmaxLayer *layer, Network *net);
void BackwardSoftmaxLayer(SoftmaxLayer *layer, Network *net);
void SoftmaxCore(float *input, int n, float temp, float *output);
void SoftmaxBatch(float *input, int n, int batch_size, float temp, float *output);
void BackwardSoftmaxCore(float *output, float *delta_output, int n, float temp, float *delta_input);
void BackwardSoftmax(float *layer_delta, int n, int batch_size, float *net_delta);

#endif