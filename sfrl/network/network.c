#include "network.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../activation/activation.h"
#include "../data/data.h"
#include "../layer/base_layer.h"
#include "../layer/batchnorm_layer.h"
#include "../loss/loss.h"
#include "../metric/metric.h"
#include "../optimizer/optimizer.h"
#include "../utils/blas.h"
#include "../utils/utils.h"

Network *MakeNetwork(int n, int batch_size) {
  Network *net = calloc(1, sizeof(Network));
  net->layer_depth = n;
  net->layers = calloc(net->layer_depth, sizeof(Layer *));
  // 默认优化参数
  net->learning_rate = 0.1;
  net->momentum = 0.9;
  net->decay = 0.1;
  net->gamma = 0.9;
  net->beta_1 = 0.9;
  net->beta_2 = 0.999;
  net->eps = 1e-8;
  net->batch_size = batch_size;
  net->simple_train = SimpleTrain;
  net->simple_test = SimpleTest;
  net->train = Train;
  net->test = Test;
  net->reset = ResetNetwork;
  net->print = PrintNetwork;
  return net;
}

void PrintNetwork(Network *net) {
  printf("net work args:\n");
  printf("batch size: %d\n", net->batch_size);
  printf("net depth : %d\n", net->layer_depth);
  int all_weight_num = 0;
  for (int i = 0; i < net->layer_depth; ++i) {
    Layer *layer = net->layers[i];
    char *layer_type_str = GetLayerTypeStr(layer->layer_type);
    char *acti_type_str = GetActivationTypeStr(layer->acti_type);
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    if (layer->layer_type == DENSE) {
      int weight_num = input_size * output_size + output_size;
      printf("-- %s: shape: %d × %d, activation: %s, weight num: %d\n", layer_type_str, input_size, output_size,
             acti_type_str, weight_num);
      all_weight_num += weight_num;
    } else if (layer->layer_type == BATCHNORMALIZATION) {
      int weight_num = output_size * 2;
      printf("-- %s: shape: %d × %d, weight num: %d\n", layer_type_str, input_size, output_size, weight_num);
      all_weight_num += weight_num;
    } else {
      printf("-- %s\n", layer_type_str);
    }
  }
  char *loss_str = GetLossStr(net->layers[net->layer_depth - 1]->loss_type);
  char *opt_str = GetOptimizerStr(net->opt_type);
  printf("-- loss: %s, optimizer: %s, all weight num: %d\n", loss_str, opt_str, all_weight_num);
}

float SimpleTrain(Network *net, Data *data, OptType opt_type, int epoches) {
  printf("epoches = %d\n", epoches);
  int batch_size = net->batch_size;
  int batch_num = data->sample_num / batch_size;
  net->mode = TRAIN;
  net->opt_type = opt_type;
  net->print(net);
  net->input_size = batch_size * data->sample_size;
  int last_batch_size = data->sample_num % batch_size;
  net->origin_input = malloc(net->input_size * sizeof(float));
  net->ground_truth = malloc(batch_size * sizeof(float));
  float loss_sum, epoch_loss = 0;
  // PrintData(data);
  for (int i = 0; i < epoches; ++i) {
    loss_sum = 0;
    net->epoch = i + 1;
    for (int j = 0; j < batch_num; ++j) {
      net->batch = j;
      //拿到一个batch的数据
      GetNextBatchData(data, net, batch_size, batch_size * j);
      net->batch_trained_cnt += batch_size;
      net->layers[0]->print_weight(net->layers[0]);
      ForwardNetwork(net);
      BackWardNetwork(net);
      UpdateNetwork(net);
      net->layers[0]->print_grad(net->layers[0]);
      ResetNetwork(net);
      loss_sum += net->loss;
    }
    if (last_batch_size) {
      // 处理最后一个batch, 要改掉所有的 batch_size
      net->batch = batch_num;
      net->batch_size = last_batch_size;
      for (int j = 0; j < net->layer_depth; ++j) {
        net->layers[j]->batch_size = last_batch_size;
      }
      // 最后一个不够 batch_size 的 batch 需要单独处理
      net->origin_input = realloc(net->origin_input, last_batch_size * data->sample_size * sizeof(float));
      net->ground_truth = realloc(net->ground_truth, last_batch_size * sizeof(float));
      GetNextBatchData(data, net, last_batch_size, batch_size * (batch_num - 2));
      net->batch_trained_cnt++;
      ForwardNetwork(net);
      BackWardNetwork(net);
      UpdateNetwork(net);
      loss_sum += net->loss;

      // batch_size 记得后面改回来
      net->batch_size = batch_size;
      for (int j = 0; j < net->layer_depth; ++j) {
        net->layers[j]->batch_size = batch_size;
      }
      net->origin_input = realloc(net->origin_input, net->input_size * sizeof(float));
      net->ground_truth = realloc(net->ground_truth, batch_size * sizeof(float));
    }
    epoch_loss = loss_sum / (batch_num * batch_size + last_batch_size);
    printf("epoch %d loss = %f\n", i + 1, epoch_loss);
    net->reset(net);
  }
  return epoch_loss;
}

float SimpleTest(Network *net, Data *data) {
  net->input_size = data->sample_num * data->sample_size;
  net->origin_input = realloc(net->origin_input, net->input_size * sizeof(float));
  net->ground_truth = realloc(net->ground_truth, data->sample_num * sizeof(float));
  GetNextBatchData(data, net, data->sample_num, 0);
  ForwardNetwork(net);
  float loss = net->loss / data->sample_num;
  float acc = AccMetric(data->sample_num, data->class_num, net->pred, net->ground_truth);
  printf("Test loss = %f, acc = %f\n", loss, acc);
  return acc;
}

float Train(Network *net, OptType opt_type, int epoches) {}

float Test(Network *net) {}

void GetNextBatchData(Data *data, Network *net, int batch_size, int offset) {
  int size = data->sample_size;
  // copy输入
  memcpy(net->origin_input, data->X + offset, data->sample_size * batch_size * sizeof(float));
  net->input = net->origin_input;
  // copy 标签
  memcpy(net->ground_truth, data->Y + offset, batch_size * sizeof(float));
}

/**
 *  正向传播，没啥好讲的
 **/
void ForwardNetwork(Network *net) {
  for (int i = 0; i < net->layer_depth; ++i) {
    net->active_layer_index = i;
    Layer *layer = net->layers[i];
    layer->ground_truth = net->ground_truth;
    layer->forward(layer, net);
    net->input = layer->output;
  }
  net->input = net->origin_input;
}

void ForwardNetworkDag(Network *net) {
  for (int i = 0; i < net->layer_depth; ++i) {
    net->active_layer_index = i;
    Layer *layer = net->layers[i];
    layer->ground_truth = net->ground_truth;
    layer->forward(layer, net);
    net->input = layer->output;
  }
  net->input = net->origin_input;
}

/**
 *  反向传播部分，维护一个delta来实现链式求导法则，delta是输入的导数
 *  delta = dL / dx
 *  dw = dL / dx * output = delta * output
 *  这里注意要先复制一份网络的输入，因为输入不是一个正常的层
 *  其他的没啥，就是上层的输出变成当前层输入(这里通过net->input维护),
 *  上一层的delta由 net->delta维护，注意啊，这个上一层是 i-1
 *  层，这里的net->delta不是参与计算 而是给它赋值，用于下一层的计算
 **/
void BackWardNetwork(Network *net) {
  for (int i = net->layer_depth - 1; i >= 0; --i) {
    Layer *layer = net->layers[i];
    net->active_layer_index = i;
    if (i > 0) {
      Layer *pre_layer = net->layers[i - 1];
      net->input = pre_layer->output;
      net->delta = pre_layer->delta;
    } else {
      // 第一层，没有前一层了
      net->input = net->origin_input;
      net->delta = NULL;
    }
    layer->backward(layer, net);
  }
}

/**
 *  通过不同的optimization方法更新网络参数
 *  没啥可说的
 **/
void UpdateNetwork(Network *net) {
  for (int i = 0; i < net->layer_depth; ++i) {
    Layer *layer = net->layers[i];
    net->active_layer_index = i;
    if (layer->update) {
      layer->update(layer, net);
    }
  }
}

void ResetNetwork(Network *net) {
  for (int i = 0; i < net->layer_depth; ++i) {
    Layer *layer = net->layers[i];
    layer->reset(layer);
  }
}

void FreeNetwork(Network *net) {
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

/**
 *  找到l1 l2 各自指针数组的空位置，然后建立 l1 <--> l2的双向链接
 **/
void LinkLayers(Layer *l1, Layer *l2) {
  assert(l1->post_layers[l1->post_layer_cnt - 1] == 0);
  assert(l2->pre_layers[l2->pre_layer_cnt - 1] == 0);
  int i = 0;
  while (l1->post_layers[++i])
    ;

  l1->post_layers[i] = l2;

  i = 0;
  while (l2->pre_layers[++i])
    ;
  l2->pre_layers[i] = l1;
}