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
  NetMode mode;
  int layer_depth;
  float epoch;
  int active_layer_index;
  float loss;
  

  // 输入输出
  float *origin_input; // 这里维护的是整个网络的输入
  float *input;  // 这里的输入维护的是当前层的输入，也就是上一层的输出
  float *output; // 这里维护的是整个网络的输出
  float *ground_truth; // 标签
  int input_size;
  int output_size;
  int batch_size;
  int sample_size; // 单个样本的size
  int input_processed_num; // 已经处理过的输入样本数量

  // 反向传播
  float *delta; // 注意这里, net->delta 只是指向反向传播时需要层的delta
  float *error;
  int batch_trained_cnt;

  // optimization 相关
  OptType opt_type;
  float decay;
  float momentum;
  float learning_rate;
  float gamma;
  float beta_1;
  float beta_2;
  float eps;
  float *grad_cum_w; // 一些优化方法中的一阶梯度累计量
  float *grad_cum_b;
  float *grad_cum_square_w; // 一些优化方法中的二阶梯度累计量
  float *grad_cum_square_b;

  float (*train)(struct NetWork *net, struct Data *data, OptType);
  float (*test)(struct NetWork *net, struct Data *data);
};

NetWork MakeNetwork(int n);
void FreeNetwork(NetWork *net);

void ForwardNetwork(NetWork *net);
void BackWardNetwork(NetWork *net);
void UpdateNetwork(NetWork *net);
void GetNextBatchData(Data *data, NetWork *net, int sample_num, int offset);
float Train(NetWork *net, Data *data, OptType opt_type);
float Test(NetWork *net, Data *data);
char *GetLayerTypeStr(LayerType layer_type);
char *GetActivationTypeStr(ActiType acti_type);
char *GetLossStr(LossType loss_type);
char *GetOptimizerStr(OptType opt_type);
void PrintNetWork(NetWork net);

#endif