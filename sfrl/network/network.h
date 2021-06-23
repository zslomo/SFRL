#ifndef NET_WORK_H
#define NET_WORK_H

#include "../activation/activation.h"
#include "../layer/base_layer.h"
#include "../optimizer/optimizer.h"
#include "../loss/loss.h"
#include "../data/data.h"
#include "../type/type.h"

struct Network {
  Layer **layers;
  NetMode mode;
  int layer_depth;
  int epoch;
  int batch;
  int active_layer_index;
  float loss;

  // 计算图确定起始点
  int start_layer_cnt;
  Layer **start_layers;
  int loss_layer_cnt;
  Layer ** loss_layers;

  // 输入输出
  float *origin_input; // 这里维护的是整个网络的输入
  float *input;  // 这里的输入维护的是当前层的输入，也就是上一层的输出
  float *output;
  float *pred;
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

  float (*simple_train)(struct Network *, struct Data *, OptType, int);
  float (*train)(struct Network *, OptType, int);
  float (*simple_test)(struct Network *, struct Data *);
  float (*test)(struct Network *, OptType, int);
  void (*reset)(struct Network *);
  void (*print)(struct Network *);
};

Network *MakeNetwork(int n, int batch_size);
void FreeNetwork(Network *net);
void ForwardNetwork(Network *net);
void BackWardNetwork(Network *net);
void UpdateNetwork(Network *net);
void GetNextBatchData(Data *data, Network *net, int sample_num, int offset);
float SimpleTrain(Network *net, Data *data, OptType opt_type, int epoches);
float Train(Network *net, OptType opt_type, int epoches);
float SimpleTest(Network *net, Data *data);
float Test(Network *net);
void PrintNetwork(Network *net);
void ResetNetwork(Network *net);
void LinkLayers(Layer *l1, Layer *l2);

#endif