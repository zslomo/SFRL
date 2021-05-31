#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "base_layer.h"
#include "../../sfrl/network/network.h"

typedef Layer SoftmaxLayer;

SoftmaxLayer MakeSoftmaxLayer(int batch_size, int input_size);
void ForwardSoftmaxLayer(const SoftmaxLayer *layer, NetWork *net);
void BackwardSoftmaxLayer(const SoftmaxLayer *layer, NetWork *net);

void BackwardSoftmaxLayer(const SoftmaxLayer *layer, NetWork *net);
void SoftmaxCore(float *input, int n, float temp, float *output);
void SoftmaxBatch(float *input, int n, int batch_size, int batch_offset, float temp, float *output);
void BackwardSoftmaxCore(float *output, float *delta_output, int n, float temp, float *delta_input);
void BackwardSoftmax(float *output, float *delta_output, int n, int batch_size, int batch_offset,
                     float temp, float *delta_input);
#endif