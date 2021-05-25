#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "sfrl/activations/activations.h"
#include "sfrl/layer/base_layer.h"
#include "sfrl/network/network.h"
#include "sfrl/utils/init.h"

typedef Layer BatchNormLayer;

BatchNormLayer MakeBatchNormLayer(int batch_size, int input_size, int output_size,
                                  ActiType acti_type, InitType init_type);
void ForwardBatchNormLayer(BatchNormLayer *layer, NetWork *net);
void BackwardBatchNormLayer(BatchNormLayer *layer, NetWork *net);
void UpdateBatchNormLayer(BatchNormLayer *layer, NetWork *net);

void BnGamaBackward(float *gamma_grads, float *delta, float *output_normed, int input_size,
                  int batch_size);
void BnBetaBackward(float *beta_grads, float *delta, int input_size, int batch_size);

#endif