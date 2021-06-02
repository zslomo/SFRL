#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../sfrl/loss/loss.h"
#include "../../sfrl/network/network.h"
#include "loss_layer.h"

LossLayer MakeLossLayer(int batch_size, int input_size, LossType loss_type) {
  LossLayer loss_layer = {0};
  loss_layer.layer_type = LOSS;

  loss_layer.batch_size = batch_size;
  loss_layer.input_size = input_size;
  loss_layer.output_size = input_size;
  loss_layer.loss_type = loss_type;

  loss_layer.output = calloc(input_size * batch_size, sizeof(float));
  loss_layer.delta = calloc(input_size * batch_size, sizeof(float));
  loss_layer.error = calloc(input_size * batch_size, sizeof(float));

  loss_layer.forward = ForwardLossLayer;
  loss_layer.backward = BackwardLossLayer;
  return loss_layer;
}

void ForwardLossLayer(LossLayer *loss_layer, NetWork *net) {
  assert(net->ground_truth);
  printf("loss start\n");
  int n = loss_layer->batch_size * loss_layer->input_size;
  // loss层是最后一层，在forward的时候记录一下整个网络的输出，用来计算metric
  net->output = net->input;
  // 计算loss
  switch (loss_layer->loss_type) {
  case MSE:
    MeanSquareError(n, net->input, net->ground_truth, loss_layer->error);
    break;
  case CE:
    CrossEntropy(n, net->input, net->ground_truth, loss_layer->error);
    break;
  case CEW:
    CrossEntropyWithWeight(n, net->input, net->ground_truth, loss_layer->error);
    break;
  default:
    break;
  }
}

void BackwardLossLayer(LossLayer *loss_layer, NetWork *net) {
  assert(net->ground_truth);
  int n = loss_layer->batch_size * loss_layer->input_size;
  switch (loss_layer->loss_type) {
  case MSE:
    BackwardMeanSquareError(n, net->input, net->ground_truth, loss_layer->delta);
    break;
  case CE:
    BackwardCrossEntropy(n, net->input, net->ground_truth, loss_layer->delta);
    break;
  case CEW:
    BackwardCrossEntropyWithWeight(n, net->input, net->ground_truth, loss_layer->delta);
    break;
  default:
    break;
  }
}