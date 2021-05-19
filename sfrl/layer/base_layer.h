#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "sfrl/activations/activations.h"
#include "sfrl/layer/cost_layer.h"
#include "sfrl/network/network.h"
#include "stddef.h"

/**
 * 网络结构类型，强化学习没有太复杂的结构,这里主要是全连接、卷积(类似棋盘游戏需要)
 *、bn 和一些激活函数 COST 是用来计算最后一步的deta也就是predict和Y的差 ACTIVE
 *表征这个层是激活函数
 * TODO 卷积、池化、RNN
 **/
typedef enum {
  DENSE,
  NORMALIZATION,
  BATCHNORMALIZATION,
  DROPOUT,
  ACTIVATION,
  SOFTMAX,
  COST
} LayerType;

/**
 * 网络层类型，比较复杂，详见每个字段的注释
 **/
struct Layer {
  LayerType layer_type;        // 层类型
  ActiType acti_type;          // 激活函数
  CostType cost_type;          // 损失函数类型

  int batch_normalize;
  // 输入输出
  float *input;
  float *output;
  int input_size;
  int output_size;
  int batch_size;

  // 计算相关
  float *deta;
  float *weights;
  float *weights_update;
  float *biases;
  float *biases_update;

  // bn相关
  float *scales;
  float *scale_updates;
  float *mean;
  float *mean_delta;
  float *variance;
  float *variance_delta;
  float *rolling_mean;
  float *rolling_variance;
  float *norm_input;
  float *norm_output;

  // dropout相关
  float probability;
  float *drop_elem;  

  // 非常重要的三个函数，分别定义了这种类型网络的前向、后向、更新操作
  void (*forward)(struct Layer, struct NetWork);
  void (*backward)(struct Layer, struct NetWork);
  void (*update)(struct Layer, int, float, float, float);
}Layer;


#endif
