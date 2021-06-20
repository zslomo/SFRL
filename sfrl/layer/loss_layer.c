#include "loss_layer.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../loss/loss.h"
#include "../network/network.h"
#include "../utils/blas.h"

LossLayer *MakeLossLayer(int batch_size, int input_size, int output_size,
                         LossType loss_type, char *layer_name) {
  LossLayer *layer = calloc(1, sizeof(LossLayer));
  layer->layer_type = LOSS;
  layer->layer_name = layer_name;
  layer->batch_size = batch_size;
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->loss_type = loss_type;
  layer->input = calloc(input_size * batch_size, sizeof(float));
  layer->output = calloc(input_size * batch_size, sizeof(float));
  layer->delta = calloc(output_size * batch_size, sizeof(float));
  layer->forward = ForwardLossLayer;
  layer->backward = BackwardLossLayer;
  layer->print_input = PrintInput;
  layer->print_output = PrintOutput;
  layer->print_delta = PrintDelta;
  layer->reset = ResetLayer;

  return layer;
}

void ForwardLossLayer(LossLayer *layer, Network *net) {
  assert(net->ground_truth);
  net->loss = 0;
  int n = net->batch_size * layer->input_size;
  net->output = net->input;
  // CopyTensor(layer->input_size * net->batch_size, net->input, layer->input);
  memcpy(layer->input, net->input, layer->input_size * net->batch_size * sizeof(float));
  InitTensor(layer->output_size * net->batch_size, 0, layer->output);
  // 计算loss
  switch (layer->loss_type) {
    case MSE:
      MeanSquareError(n, net->input, net->ground_truth, layer->output);
      break;
    case CE:
      CrossEntropy(net->batch_size, layer->input_size, net->input,
                   net->ground_truth, layer->output, 0);
      break;
    case CEW:
      CrossEntropy(net->batch_size, layer->input_size, net->input,
                   net->ground_truth, layer->output, 1);
      break;
    default:
      break;
  }
  // printf("epoch %d, batch %d, loss: ", net->epoch, net->batch);
  for (int i = 0; i < net->batch_size; ++i) {
    float tmp = 0;
    for (int j = 0; j < layer->input_size; ++j) {
      tmp += layer->output[i * layer->input_size + j];
    }
    // printf("%f,", tmp);
    net->loss += tmp;
  }
  // printf("\n");
}

void BackwardLossLayer(LossLayer *layer, Network *net) {
  assert(net->ground_truth);
  int n = net->batch_size * layer->input_size;
  switch (layer->loss_type) {
    case MSE:
      BackwardMeanSquareError(n, net->input, net->ground_truth, layer->delta);
      break;
    case CE:
      BackwardCrossEntropy(net->batch_size, layer->input_size, net->input,
                           net->ground_truth, layer->delta, 0);
      break;
    case CEW:
      BackwardCrossEntropy(net->batch_size, layer->input_size, net->input,
                           net->ground_truth, layer->delta, 1);
      break;
    default:
      break;
  }
  memcpy(net->delta, layer->delta, n * sizeof(float));
}