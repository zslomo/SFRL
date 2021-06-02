#include "network.h"
#include "../../sfrl/activation/activation.h"
#include "../../sfrl/data/data.h"
#include "../../sfrl/layer/base_layer.h"
#include "../../sfrl/layer/batchnorm_layer.h"
#include "../../sfrl/optimizer/optimizer.h"
#include "../../sfrl/utils/blas.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  net.layer_depth = n;
  net.layers = calloc(net.layer_depth, sizeof(Layer *));
  // 默认优化参数
  net.learning_rate = 0.1;
  net.momentum = 0.9;
  net.decay = 0.1;
  net.gamma = 0.9;
  net.beta_1 = 0.9;
  net.beta_2 = 0.999;
  net.eps = 1e-8;
  net.train = Train;
  net.test = Test;
  return net;
}

char *GetLayerTypeStr(LayerType layer_type) {
  char *layer_type_str;
  if (layer_type == DENSE) {
    layer_type_str = "Dense";
  } else if (layer_type == BATCHNORMALIZATION) {
    layer_type_str = "BatchNorm";
  } else if (layer_type == SOFTMAX) {
    layer_type_str = "SoftMax";
  } else if (layer_type == DROPOUT) {
    layer_type_str = "DropOut";
  }else if (layer_type == ACTIVATION) {
    layer_type_str = "Activation";
  }else if (layer_type == LOSS) {
    layer_type_str = "Loss";
  } else {
    layer_type_str = "error";
  }
  return layer_type_str;
}

char *GetActivationTypeStr(ActiType acti_type) {
  char *acti_type_str;
  if (acti_type == LINEAR) {
    acti_type_str = "linear";
  } else if (acti_type == SIGMOID) {
    acti_type_str = "sigmoid";
  } else if (acti_type == RELU) {
    acti_type_str = "relu";
  } else if (acti_type == TANH) {
    acti_type_str = "tanh";
  } else {
    acti_type_str = "error";
  }
  return acti_type_str;
}

char *GetLossStr(LossType loss_type) {
  char *loss_str;
  if (loss_type == MSE) {
    loss_str = "MeanSquareError";
  } else if (loss_type == CE) {
    loss_str = "CrossEntropy";
  } else if (loss_type == CEW) {
    loss_str = "CrossEntropyWeight";
  } else {
    loss_str = "error";
  }
  return loss_str;
}

char *GetOptimizerStr(OptType opt_type) {
  char *opt_str;
  if (opt_type == ADAM) {
    opt_str = "adam";
  } else if (opt_type == SGD) {
    opt_str = "sgd";
  } else if (opt_type == ADAGRAD) {
    opt_str = "adagrad";
  } else if (opt_type == RMSPROP) {
    opt_str = "rmsprop";
  } else {
    opt_str = "error";
  }
  return opt_str;
}

void PrintNetWork(NetWork net) {
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

float Train(NetWork *net, Data *data, OptType opt_type) {
  int batch_size = net->batch_size;
  int batch_num = data->sample_num / batch_size;
  net->mode = TRAIN;
  net->opt_type = opt_type;
  float sum = 0;
  PrintNetWork(*net);
  net->input_size = batch_size * data->sample_size;
  net->origin_input = malloc(net->input_size * sizeof(float));
  net->ground_truth = malloc(batch_size * sizeof(float));
  for (int i = 0; i < batch_num - 1; ++i) {
    // 拿到一个batch的数据
    printf("batch %d start...\n", i);
    GetNextBatchData(data, net, batch_size, batch_size * i);
    printf("batch %d get data done.\n", i);
    net->batch_trained_cnt += batch_size;
    ForwardNetwork(net);
    printf("batch %d forward done.\n", i);
    BackWardNetwork(net);
    float *tmp = calloc(32, sizeof(float));
    printf("batch %d backward done.\n", i);
    UpdateNetwork(net);
    printf("batch %d update done.\n", i);
    sum += net->loss;
    printf("batch %d done.\n", i);
  }
  // 处理最后一个batch
  // 要改掉所有的 batch_size
  int last_batch_size = data->sample_num % batch_size;
  net->batch_size = last_batch_size;
  for (int i = 0; i < net->layer_depth; ++i) {
    net->layers[i]->batch_size = last_batch_size;
  }
  // 最后一个不够 batch_size 的 batch 需要单独处理
  net->input = realloc(net->input, last_batch_size * data->sample_size * sizeof(float));
  net->ground_truth = realloc(net->ground_truth, last_batch_size * sizeof(float));
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

float Test(NetWork *net, Data *data) {
  net->mode = TEST;
  int batch_size = data->sample_num;
  int batch_num = 1;
  net->input = malloc(batch_size * data->sample_size * sizeof(float));
  net->ground_truth = malloc(batch_size * sizeof(float));
  GetNextBatchData(data, net, batch_size, 0);
  ForwardNetwork(net);
  return net->loss;
}

void GetNextBatchData(Data *data, NetWork *net, int sample_num, int offset) {
  int size = data->sample_size;
  // printf("sample_num = %d, offset = %d, size = %d\n", sample_num, offset, size);
  // copy输入
  memcpy(net->origin_input, data->X + offset, size * sample_num * sizeof(float));
  net->input = net->origin_input;
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
  printf("loss: ");
  for (int i = 0; i < net->batch_size; ++i) {
    printf("%f,", net->input[i]);
  }
  printf("\n"); 
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
  for (int i = net->layer_depth - 1; i >= 0; --i) {
    Layer *layer = net->layers[i];
    if (i != 0) {
      Layer *pre_layer = net->layers[i - 1];
      net->input = pre_layer->output;
      net->delta = pre_layer->delta;
      float *tmp = calloc(32, sizeof(float));
    } else {
      // 第一层，输入就是网络的输入
      net->input = net->origin_input;
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
    printf("layer %d update done\n", i);
  }
}