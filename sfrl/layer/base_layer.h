#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "sfrl/activations/activations.h"
#include "sfrl/layer/loss_layer.h"
#include "sfrl/network/network.h"
#include "stddef.h"

/**
 * 网络结构类型，强化学习没有太复杂的结构,这里主要是全连接、卷积(类似棋盘游戏需要)
 * bn 和一些激活函数 LOSS 是用来计算最后一步的deta也就是predict和Y的差
 * ACTIVE 表征这个层是激活函数
 * TODO 卷积、池化、RNN
 **/
typedef enum {
  DENSE,
  NORMALIZATION,
  BATCHNORMALIZATION,
  DROPOUT,
  ACTIVATION,
  SOFTMAX,
  LOSS
} LayerType;

/**
 * 网络层类型，比较复杂，详见每个字段的注释
 **/
struct Layer {
  LayerType layer_type; // 层类型
  ActiType acti_type;   // 激活函数
  LossType loss_type;   // 损失函数类型

  int batch_normalize;
  /**
   * 输入输出
   * 这里是个值得探讨的地方，在darnet的实现中 layer 是不维护input只维护output的，network结构中会维护当前层的input
   * 其实维护input是一个很冗余的事情：
   *   1 上一层的output跟下一层的 input一样，所以维护两个一样的数据意义不大，通过指针链接两个层其实可以方便的拿到上层的输出
   *   2 对于网络来说，输入输出没有什么意义，只是在更新参数，理论上只需要维护当前活跃层的输入输出即可，可以节省很多空间
   * */
  // float *input;
  float *output;
  int input_size;
  int output_size;
  int batch_size;
  int group_size;

  // 计算相关
  float loss;              // 只有最后一层有计算loss
  float *deta;             // 误差函数关于当前层每个加权输入的导数值 用来求权重的导数,导数 = deta[i] * output[i]
  float *weights;      
  float *weight_updates;   // 权重更新值，反向传播的导数
  float *biases;
  float *bias_updates;     // 偏置更新值，反向传播的导数

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

  // softmax 相关
  float temperature;

  // dropout相关
  float probability;
  float *drop_elem;

  // 激活层相关

  // 非常重要的三个函数，分别定义了这种类型网络的前向、后向、更新操作
  void (*forward)(struct Layer, struct NetWork);
  void (*backward)(struct Layer, struct NetWork);
  void (*update)(struct Layer, int, float, float, float);
} Layer;

#endif
