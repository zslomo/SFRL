#include "network.h"
#include "../activation/activation.h"
#include "../data/data.h"
#include "../layer/base_layer.h"
#include "../layer/batchnorm_layer.h"
#include "../loss/loss.h"
#include "../optimizer/optimizer.h"
#include "../utils/blas.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Network MakeNetwork(int n) {
  Network net = {0};
  net.layer_depth = n;
  net.layers = calloc(net.layer_depth, sizeof(Layer *));
  // 默认优化参数
  net.learning_rate = 0.01;
  net.momentum = 0.9;
  net.decay = 0.1;
  net.gamma = 0.9;
  net.beta_1 = 0.9;
  net.beta_2 = 0.999;
  net.eps = 1e-8;
  net.train = Train;
  net.test = Test;
  net.reset = ResetNetwork;

  return net;
}

void PrintNetwork(Network net) {
  printf("net work args:\n");
  printf("batch size: %d\n", net.batch_size);
  printf("net depth : %d\n", net.layer_depth);
  int all_weight_num = 0;
  for (int i = 0; i < net.layer_depth - 1; ++i) {
    Layer *layer = net.layers[i];
    char *layer_type_str = GetLayerTypeStr(layer->layer_type);
    char *acti_type_str = GetActivationTypeStr(layer->acti_type);
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int weight_num = input_size * output_size + output_size;
    if (layer->layer_type == DENSE || layer->layer_type == BATCHNORMALIZATION) {
      printf("-- %s: shape: %d × %d, activation: %s, weight num: %d\n", layer_type_str, input_size,
             output_size, acti_type_str, weight_num);
      all_weight_num += weight_num;
    } else {
      printf("-- %s\n", layer_type_str);
    }
  }
  char *loss_str = GetLossStr(net.layers[net.layer_depth - 1]->loss_type);
  char *opt_str = GetOptimizerStr(net.opt_type);
  printf("-- loss: %s, optimizer: %s, all weight num: %d\n", loss_str, opt_str, all_weight_num);
}

float Train(Network *net, Data *data, OptType opt_type, int epoches) {
  printf("epoches = %f\n", epoches);
  int batch_size = net->batch_size;
  int batch_num = data->sample_num / batch_size;
  net->mode = TRAIN;
  net->opt_type = opt_type;
  PrintNetwork(*net);
  net->input_size = batch_size * data->sample_size;
  net->origin_input = malloc(net->input_size * sizeof(float));
  net->ground_truth = malloc(batch_size * sizeof(float));
  float sum, epoch_loss = 0;
  // PrintData(data);
  for (int i = 0; i < epoches; ++i) {
    sum = 0;
    net->epoch = i + 1;
    for (int j = 0; j < batch_num - 1; ++j) {
      net->batch = j;
      printf("--------------------------- epoch %d, batch %d start --------------------------\n", i,
             j);
      // 拿到一个batch的数据
      GetNextBatchData(data, net, batch_size, batch_size * j);
      printf("batch %d get data done.\n", j);
      net->batch_trained_cnt += batch_size;
      ForwardNetwork(net);
      net->layers[2]->print_input(net->layers[2], 4);
      // net->layers[2]->print_weight(net->layers[2]);
      net->layers[2]->print_output(net->layers[2], 4);
      // net->layers[0]->print_grad(net->layers[0]);
      // printf("batch %d forward done.\n", j);
      BackWardNetwork(net);
      net->layers[2]->print_delta(net->layers[2], 4);
      // printf("batch %d backward done.\n", j);
      UpdateNetwork(net);
      // net->layers[2]->print_update(net->layers[2]);
      // printf("batch %d update done.\n", j);
      sum += net->loss;
      // printf("batch %d done. loss = %f\n", j, net->loss/net->batch_size);
    }
    // 处理最后一个batch
    // 要改掉所有的 batch_size
    net->batch = batch_num;
    int last_batch_size = data->sample_num % batch_size;
    net->batch_size = last_batch_size;
    for (int j = 0; j < net->layer_depth; ++j) {
      net->layers[j]->batch_size = last_batch_size;
    }
    // 最后一个不够 batch_size 的 batch 需要单独处理
    net->origin_input =
        realloc(net->origin_input, last_batch_size * data->sample_size * sizeof(float));
    net->ground_truth = realloc(net->ground_truth, last_batch_size * sizeof(float));
    GetNextBatchData(data, net, last_batch_size, batch_size * (batch_num - 2));
    net->batch_trained_cnt++;
    ForwardNetwork(net);
    BackWardNetwork(net);
    UpdateNetwork(net);
    sum += net->loss;

    // batch_size 记得后面改回来
    net->batch_size = batch_size;
    for (int j = 0; j < net->layer_depth; ++j) {
      net->layers[j]->batch_size = batch_size;
    }
    net->origin_input = realloc(net->origin_input, net->input_size * sizeof(float));
    net->ground_truth = realloc(net->ground_truth, batch_size * sizeof(float));

    epoch_loss = sum / (batch_num * batch_size + last_batch_size);
    printf("epoch %d loss = %f\n", i + 1, epoch_loss);
    net->reset(net);
  }
  return epoch_loss;
}

float Test(Network *net, Data *data) {
  net->mode = TEST;
  int batch_size = data->sample_num;
  int batch_num = 1;
  net->input = malloc(batch_size * data->sample_size * sizeof(float));
  net->ground_truth = malloc(batch_size * sizeof(float));
  GetNextBatchData(data, net, batch_size, 0);
  ForwardNetwork(net);
  return net->loss;
}

void GetNextBatchData(Data *data, Network *net, int sample_num, int offset) {
  int size = data->sample_size;
  // copy输入
  memcpy(net->origin_input, data->X + offset, data->sample_size * sample_num * sizeof(float));
  net->input = net->origin_input;
  // copy 标签
  memcpy(net->ground_truth, data->Y + offset, sample_num * sizeof(float));
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