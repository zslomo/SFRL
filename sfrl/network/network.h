#ifndef NET_WORK_H
#define NET_WORK_H

#include "sfrl/activations/activations.h"
#include "sfrl/layer/base_layer.h"

typedef struct NetWork {
  int layer_depth;
  float epoch;
  int active_layer_index;
  float *cost;

  // 输入输出
  float *input;
  float *output;
  int input_size;
  int output_size;
  int batch_size;
  float *delta;

  // 网络空间
  float *workspace;

  // train test
  int mode;

  // 计算相关
  float decay;
  float momentum;
  float learning_rate;
  float gamma;
  float scale;

  // optimization 相关
  float B1;
  float B2;
  float eps;

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