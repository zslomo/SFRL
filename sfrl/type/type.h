#ifndef TYPE_H
#define TYPE_H

// 激活函数
typedef enum { SIGMOID, RELU, LINEAR, TANH } ActiType;

/**
 * 网络结构类型，强化学习没有太复杂的结构,这里主要是全连接、卷积(类似棋盘游戏需要)
 * bn 和一些激活函数 LOSS 是用来计算最后一步的delta也就是predict和Y的差
 * ACTIVE 表征这个层是激活函数
 * TODO 卷积、池化、RNN
 **/
typedef enum {
  DENSE,
  NORMALIZATION,
  BATCHNORMALIZATION,
  DROPOUT,
  MERGE,
  ACTIVATION,
  SOFTMAX,
  LOSS
} LayerType;

/**
 * 损失函数
 **/
typedef enum { MSE, CE, CEW } LossType;

/**
 * metric
 **/
typedef enum { MSEM, ACC } MetricType;

/**
 * 初始化类型
 **/
typedef enum { NORMAL, UNIFORM, DEBUG } InitType;

/**
 * net 类型
 **/
typedef enum { TRAIN, TEST } NetMode;

/**
 * 优化 类型
 **/
typedef enum { ADAM, SGD, ADAGRAD, RMSPROP } OptType;

/**
 * merge 类型
 **/
typedef enum { SUM, DOT, AVG, CONCAT } MergeType;

#endif