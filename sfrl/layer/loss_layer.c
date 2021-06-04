#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../loss/loss.h"
#include "../network/network.h"
#include "loss_layer.h"

LossLayer MakeLossLayer(int batch_size, int input_size, LossType loss_type) {
  LossLayer layer = {0};
  layer.layer_type = LOSS;

  layer.batch_size = batch_size;
  layer.input_size = input_size;
  layer.output_size = input_size;
  layer.loss_type = loss_type;

  layer.input = calloc(input_size * batch_size, sizeof(float));
  layer.output = calloc(input_size * batch_size, sizeof(float));
  layer.delta = calloc(input_size * batch_size, sizeof(float));

  layer.forward = ForwardLossLayer;
  layer.backward = BackwardLossLayer;
  layer.print_input = PrintInput;
  layer.print_output = PrintOutput;
  layer.print_delta = PrintDelta;
  layer.reset = ResetLayer;

  return layer;
}

void ForwardLossLayer(LossLayer *layer, Network *net) {
  assert(net->ground_truth);
  int n = net->batch_size * layer->input_size;
  // loss层是最后一层，在forward的时候记录一下整个网络的输出，用来计算metric
  net->output = net->input;
  memcpy(layer->input, net->input, layer->input_size * layer->batch_size);
  // printf("before loss output : \n");
  // for(int i = 0; i < n; i++){
  //   printf("%f ", net->input[i]);
  // }
  // printf("\n");
  // 计算loss
  switch (layer->loss_type) {
  case MSE:
    MeanSquareError(n, net->input, net->ground_truth, layer->output);
    break;
  case CE:
    CrossEntropy(net->batch_size, layer->input_size, net->input, net->ground_truth, layer->output,
                 0);
    break;
  case CEW:
    CrossEntropy(net->batch_size, layer->input_size, net->input, net->ground_truth, layer->output,
                 1);
    break;
  default:
    break;
  }
  printf("loss: ");
  for (int i = 0; i < net->batch_size; ++i) {
    printf("%0.8f,", layer->output[i]);
    net->loss += layer->output[i];
  }
  net->loss /= net->batch_size;
  printf("\n");
}

void BackwardLossLayer(LossLayer *layer, Network *net) {
  assert(net->ground_truth);
  int n = net->batch_size * layer->input_size;
  switch (layer->loss_type) {
  case MSE:
    BackwardMeanSquareError(n, net->input, net->ground_truth, layer->delta);
    break;
  case CE:
    BackwardCrossEntropy(net->batch_size, layer->input_size, net->input, net->ground_truth,
                         layer->delta, 0);
    break;
  case CEW:
    BackwardCrossEntropy(net->batch_size, layer->input_size, net->input, net->ground_truth,
                         layer->delta, 1);
    break;
  default:
    break;
  }
  memcpy(net->delta, layer->delta, n * sizeof(float));
}