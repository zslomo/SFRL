#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H
#include "../activation/activation.h"
#include "../network/network.h"
#include "../utils/init.h"
#include "base_layer.h"

/**
 *  BN
 * 是放在激活函数之前还是激活函数之后有争议，原论文是放在激活函数之前的，目前有些研究表明BN放在激活函数之后效果更好
 *  这里选择放在激活函数之后，原因是因为懒，这样BN就作为一个单独的层存在，否则需要吧BN的操作写在每个层内部，会非常繁琐
 *  就是这样，XD
 * */
typedef Layer BatchNormLayer;

BatchNormLayer MakeBatchNormLayer(int batch_size, int input_size, ActiType acti_type,
                                  InitType init_type);
void ForwardBatchNormLayer(BatchNormLayer *layer, NetWork *net);
void BackwardBatchNormLayer(BatchNormLayer *layer, NetWork *net);

void BnGamaBackward(float *delta, float *output_normed, int input_size, int batch_size,
                    float *gamma_grads);
void BnBetaBackward(float *delta, int input_size, int batch_size, float *beta_grads);
void BnDot(float *gamma, int input_size, int batch_size, float *delta);
void BnMeanDelta(float *variance, float *delta, float *gamma, int input_size, int batch_size,
                 float *mean_delta);
void BnVaianceDelta(float *variance_delta, float *output_before_norm, float *delta, float *mean,
                    float *variance, int input_size, int batch_size);
void BnNormDelta(float *output_before_norm, float *mean, float *variance, float *mean_delta,
                 float *variance_delta, int input_size, int batch_size, float *delta);

#endif