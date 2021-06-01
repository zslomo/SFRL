#ifndef NET_WORK_H
#define NET_WORK_H

#include "../../sfrl/activation/activation.h"
#include "../../sfrl/layer/base_layer.h"
#include "../../sfrl/optimizer/optimizer.h"
#include "../../sfrl/loss/loss.h"
#include "../../sfrl/data/data.h"

typedef enum { TRAIN, TEST } NetMode;

struct NetWork {
  Layer **layers;
  int layer_depth;
  float epoch;
  int active_layer_index;
  float loss;
  NetMode mode;

  // 输入输出
  float *input; // 这里的输入维护的是当前层的输入，也就是上一层的输出
  float *output; // 这里维护的是整个网络的输出
  int input_size;
  int output_size;
  int batch_size;
  int input_processed_num; // 已经处理过的输入样本数量

  // 反向传播
  float *delta; // 注意这里, net->delta 只是指向反向传播时需要层的delta
  float *error;
  // 网络空间
  float *workspace; // 架构是参考了darknet，所以这里设置了一个暂存空间
  int batch_trained_cnt;

  // optimization 相关
  OptType opt_type;
  float decay;
  float momentum;
  float learning_rate;
  float gamma;
  float scale;
  float beta_1;
  float beta_2;
  float eps;
  float *grad_cum_w; // 一些优化方法中的一阶梯度累计量
  float *grad_cum_b;
  float *grad_cum_square_w; // 一些优化方法中的二阶梯度累计量
  float *grad_cum_square_b;

  //标签
  int ground_truth_size;
  float *ground_truth;

};

NetWork MakeNetwork(int n);
void FreeNetwork(NetWork *net);

void ForwardNetwork(NetWork *net);
void BackWardNetwork(NetWork *net);
void UpdateNetwork(NetWork *net);
void GetNextBatchData(Data *data, NetWork *net, int sample_num, int offset);
float TrainNetwork(NetWork *net, Data *data);

#endif