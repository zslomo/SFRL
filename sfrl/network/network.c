#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "network.h"
#include "../../sfrl/activation/activation.h"
#include "../../sfrl/data/data.h"
#include "../../sfrl/layer/base_layer.h"
#include "../../sfrl/layer/batchnorm_layer.h"
#include "../../sfrl/optimizer/optimizer.h"
#include "../../sfrl/utils/blas.h"

void FreeNetwork(NetWork *net) {
  for (int i = 0; i < net->layer_depth; ++i) {
    FreeLayer(net->layers[i]);
  }
  free(net->layers);
  if (net->input) {
    free(net->input);
  }
  if (net->output) {
    free(net->output);
  }
  if (net->ground_truth) {
    free(net->ground_truth);
  }
}

NetWork MakeNetwork(int n) {
  NetWork net = {0};
  net.layers = calloc(net.layer_depth, sizeof(Layer *));
  return net;
}

float Train(NetWork *net, Data *data) {
  int loss_avg = 0;
  int batch_size = net->batch_size;
  int batch_num = data->batch / batch_size;
  net->mode = TRAIN;
  float sum = 0;
  for (int i = 0; i < batch_num - 1; ++i) {
    // 拿到一个batch的数据
    GetNextBatchData(data, net, batch_size, batch_size * i);
    net->batch_trained_cnt += batch_size;
    ForwardNetwork(net);
    BackWardNetwork(net);
    UpdateNetwork(net);
    sum += net->loss;
  }
  // 处理最后一个batch
  // 要改掉所有的 batch_size
  int last_batch_size = data->last_batch;
  net->batch_size = last_batch_size;
  for (int i = 0; i < net->layer_depth; ++i) {
    net->layers[i]->batch_size = last_batch_size;
  }
  // 最后一个不够 batch_size 的 batch 需要单独处理
  GetNextBatchData(data, net, last_batch_size, batch_size * (batch_num - 2));
  net->batch_trained_cnt++;
  ForwardNetwork(net);
  BackWardNetwork(net);
  UpdateNetwork(net);
  sum += net->loss;

  // batch_size 记得后面改回来
  net->batch_size = batch_size;
  for (int i = 0; i < net->layer_depth; ++i) {
    net->layers[i]->batch_size = batch_size;
  }

  return sum / (batch_num * batch_size + last_batch_size);
}

float Test(NetWork *net, Data *data) { net->mode = TEST; }

void GetNextBatchData(Data *data, NetWork *net, int sample_num, int offset) {
  int size = data->size_per_sample;
  // copy输入
  for (int j = 0; j < sample_num; ++j) {
    memcpy(net->input, data->X + offset * j, size * sizeof(float));
  }
  // copy 标签
  memcpy(net->ground_truth, data->Y + offset, size * sizeof(float));
}

/**
 *  正向传播，没啥好讲的
 **/
void ForwardNetwork(NetWork *net) {
  for (int i = 0; i < net->layer_depth; ++i) {
    net->active_layer_index = i;
    Layer *layer = (net->layers[i]);
    layer->forward(layer, net);
    // layer 是没有 input这个成员变量的，当前层的输入就是上一层的输出
    // 所以没有必要存两份，这里直接让net->input 指向上一层的输出，当做当前层的输入就好了
    net->input = layer->output;
  }
}
/**
 *  反向传播部分，维护一个delta来实现链式求导法则，delta是输入的导数
 *  delta = dL / dx
 *  dw = dL / dx * output = delta * output
 *  这里注意要先复制一份网络的输入，因为输入不是一个正常的层
 *  其他的没啥，就是上层的输出变成当前层输入(这里通过net->input维护),
 *  上一层的delta由 net->delta维护，注意啊，这个上一层是 i-1 层，这里的net->delta不是参与计算
 *  而是给它赋值，用于下一层的计算
 **/
void BackWardNetwork(NetWork *net) {
  // 先暂存一下整个网络的输入
  int net_input_size = net->layers[0]->input_size;
  float *net_input = calloc(net_input_size, sizeof(float));
  memcpy(net_input, net->input, net_input_size * sizeof(float));

  for (int i = net->layer_depth - 1; i >= 0; --i) {
    Layer *layer = net->layers[i];
    if (i != 0) {
      Layer *pre_layer = net->layers[i - 1];
      net->input = pre_layer->output;
      net->delta = pre_layer->delta;
    } else {
      free(net->input);
      net->input = calloc(net_input_size, sizeof(float));
      memcpy(net->input, net_input, net_input_size * sizeof(float));
      free(net_input);
    }
    net->active_layer_index = i;
    layer->backward(layer, net);
  }
}
/**
 *  通过不同的optimization方法更新网络参数
 *  没啥可说的
 **/
void UpdateNetwork(NetWork *net) {
  for (int i = 0; i < net->layer_depth; ++i) {
    Layer *layer = net->layers[i];
    if (layer->update) {
      layer->update(layer, net);
    }
  }
}