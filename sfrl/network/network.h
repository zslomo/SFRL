#ifndef NET_WORK_H
#define NET_WORK_H

#include "sfrl/activations/activations.h"
#include "sfrl/optimizer/optimizer.h"
#include "sfrl/layer/base_layer.h"

typedef enum { TRIAN, TEST } NetMode;

typedef struct NetWork {
  int layer_depth;
  float epoch;
  int active_layer_index;
  float *cost;
  NetMode mode;

  // 输入输出
  float *input;
  float *output;
  int input_size;
  int output_size;
  int batch_size;
  float *delta;        // 反向传播时上一层(i+1)层的delta 是计算当前层delta的输入值

  // 网络空间
  float *workspace;  

  // optimization 相关
  OptType opt_type;
  float decay;
  float momentum;
  float learning_rate;
  float gamma;
  float scale;
  float B1;
  float B2;
  float eps;
  float *grad_cum_w               // 一些优化方法中的一阶梯度累计量
  float *grad_cum_b
  float *grad_cum_square_w        // 一些优化方法中的二阶梯度累计量
  float *grad_cum_square_b

  //标签
  int ground_truth_size;
  float *ground_truth;

}NetWork;

NetWork MakeNetwork(int n);
void ForwardNetwork(NetWork net);
void BackWardNetwork(NetWork net);
void UpdateNetwork(NetWork net);

float TrainNetwork(NetWork net, data d);
#endif