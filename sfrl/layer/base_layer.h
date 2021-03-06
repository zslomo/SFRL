#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "../activation/activation.h"
#include "../loss/loss.h"
#include "../type/type.h"
#include <stddef.h>
#include <pthread.h>

struct Layer;
typedef struct Layer Layer;

struct Network;
typedef struct Network Network;

/**
 * 网络层类型，比较复杂，详见每个字段的注释
 **/
struct Layer {
  LayerType layer_type; // 层类型
  ActiType acti_type;   // 激活函数
  LossType loss_type;   // 损失函数类型
  MergeType merge_type; // merge类型
  char *layer_name;
  float loss_weight;
  /**
   *  这两个指针数组用来指向当前层的上n层和下n层
   *  通过这种设计，构建一个layer graph，类似于tensorflow的计算图，但是简单很多，
   *  tf是通过对每个op构造计算图，可以支持任意构造的计算流，也就天然的支持任意复杂的计算逻辑，
   *  并可以做到非常细致的优化，抛开易用性，tf的设计理念是非常棒的，只是产品做不做的好跟技术
   *  往往无关，在用户体验方面被pytorch全面超越，tf2全面重做也于事无补，一声叹息啊
   *  注意：
   *  这里设计中 多输入合并操作只允许在merge层完成，那么其他层不需要 pre_layers 字段
   *  另外 loss是最后一层，显然是不允许存在 post_layers 字段
   **/
  Layer **pre_layers;
  int pre_layer_cnt;
  Layer **post_layers;
  int post_layer_cnt;

  int batch_normalize;
  /**
   *  输入输出
   *  这里是个值得探讨的地方，在darnet的实现中 layer
   *  是不维护input只维护output的，network结构中会维护当前层的input
   * 其实维护input是一个很冗余的事情： 1 上一层的output跟下一层的
   *  input一样，所以维护两个一样的数据意义不大，通过指针链接两个层其实可以方便的拿到上层的输出 2
   *  对于网络来说，输入输出没有什么意义，只是在更新参数，理论上只需要维护当前活跃层的输入输出即可，可以节省很多空间
   *  But !
   *  But!!!!!!!!
   *  随时获取网络的输入 输出 权重 梯度
   *  是debug非常需要的东西，所以这里还是给了一个指针，保存每层的输入
   * */
  float *input;
  float *output;
  int input_size;
  int output_size;
  int batch_size;
  int group_size;
  float *ground_truth;

  // 计算相关
  /**
   *  delta 是实现链式法则的核心，代表误差函数关于当前层每个加权输入的导数值
   *  用来求权重的导数 dL/dx = dL/ddelta[i] *output[i]
   * */
  float *delta;    
  float *weights;
  float *weight_grads; // 权重更新值，反向传播的导数
  float *weight_updates;
  float *biases;
  float *bias_grads; // 偏置更新值，反向传播的导数
  float *bias_updates;

  // bn相关
  float *bn_gammas;
  float *bn_gamma_grads;
  float *bn_betas;
  float *bn_beta_grads;
  float *mean;
  float *mean_delta;
  float *variance;
  float *variance_delta;
  float *rolling_mean;
  float *rolling_variance;
  float rolling_momentum;
  float *output_normed; // 存储一下norm前后的输出值
  float *output_before_norm;

  // softmax 相关
  float temperature;

  // dropout相关
  float probability;
  float *drop_elem;

  // optimizer 相关
  float *grad_cum_w; // 一些优化方法中的一阶梯度累计量
  float *grad_cum_b;
  float *grad_cum_square_w; // 一些优化方法中的二阶梯度累计量
  float *grad_cum_square_b;

  /**
   *  非常重要的三个函数，分别定义了这种类型网络的前向、后向、更新操作
   *  这里要注意，只有struct中的指针成员变量可以被改变，其他的成员变量都只能局部生效
   **/
  void (*forward)(struct Layer *, struct Network *);
  /**
   * 前向计算每个神经元的输出值 a_j j表示网络的第j个神经元
   * 
   * 
   * */
  void (*backward)(struct Layer *, struct Network *);
  void (*update)(struct Layer *, struct Network *);
  void (*print_weight)(struct Layer *);
  void (*print_input)(struct Layer *, int);
  void (*print_output)(struct Layer *, int);
  void (*print_grad)(struct Layer *);
  void (*print_delta)(struct Layer *, int);
  void (*print_update)(struct Layer *);
  void (*reset)(struct Layer *);
};

void UpdateLayer(Layer *layer, Network *network);
void FreeLayer(Layer *layer);
void PrintWeight(Layer *layer);
void PrintInput(Layer *layer, int batch_num);
void PrintOutput(Layer *layer, int batch_num);
void PrintGrad(Layer *layer);
void PrintDelta(Layer *layer, int batch_num);
void PrintUpdate(Layer *layer);
void ResetLayer(Layer *layer);

#endif
