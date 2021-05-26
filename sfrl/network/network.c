#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "sfrl/activations/activations.h"
#include "sfrl/layer/base_layer.h"
#include "sfrl/layer/batchnorm_layer.h"
#include "sfrl/network/network.h"
#include "sfrl/optimizer/optimizer.h"
#include "sfrl/utils/blas.h"

void FreeNetwork(network *net) {
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
  net->layers = calloc(net->layer_depth, sizeof(Layer));
  return net;
}

void ForwardNetwork(NetWork *net) {
  for (int i = 0; i < net->layer_depth; ++i) {
    net->active_layer_index = i;
    Layer layer = net->layers[i];
    layer.forward(&layer, &net);
    // layer 是没有 input这个成员变量的，当前层的输入就是上一层的输出
    // 所以没有必要存两份，这里直接让net->input 指向上一层的输出，当做当前层的输入就好了
    net->input = layer->output;
  }
}

void BackWardNetwork(NetWork *net) {
  // 先暂存一下整个网络的输入
  int net_input_size = net->layers[0]->input_size;
  float *net_input = calloc(net_input_size, sizeof(float));
  memcpy(net->layers[0]->inputs, net_input, net_input_size * sizeof(float));

  for (int i = net->layer_depth - 1; i >= 0; --i) {
    Layer *layer = net->layers[i];
    if (i != 0) {
      Layer *pre_layer = net->layers[i - 1];
      net->input = pre_layer->output;
      net->delta = pre_layer->delta;
    } else {
      free(net->input);
      net->input = calloc(net_input_size, sizeof(float));
      memcpy(net_input, net->input, net_input_size * sizeof(float));
      free(net_input);
    }
    net->index = i;
    layer->BackWardNetwork(&layer, &net);
  }
}