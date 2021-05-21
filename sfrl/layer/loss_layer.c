#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "sfrl/layer/loss_layer.h"
#include "sfrl/network/network.h"
#include "sfrl/loss/loss.h"

LossLayer MakeLossLayer(int batch_size, int input_size, LossType loss_type,
                        float scale) {
  LossLayer loss_layer = {0};
  loss_layer.layer_type = LOSS;

  loss_layer.scale = scale;
  loss_layer.batch_size = batch_size;
  loss_layer.input_size = input_size;
  loss_layer.output_size = input_size;
  loss_layer.loss_type = loss_type;

  loss_layer.output = calloc(input_size * batch_size, sizeof(float));
  loss_layer.deta = calloc(input_size * batch_size, sizeof(float));
  loss_layer.loss = 0;

  loss_layer.forward = ForwardLossLayer;
  loss_layer.backward = BackwardLossLayer;
}

ForwardLossLayer(LossLayer loss_layer, NetWork net) {
  assert(net.ground_truth);
  switch (loss_layer.loss_type)
  {
  case MSE:
      int n = loss_layer.batch_size * loss_layer.input_size;
      Mse(n, net.input, net.ground_truth, loss_layer.delta, loss_layer.error);
      break;
  case SOFTMAX:
      int n = loss_layer.batch_size * loss_layer.input_size;
      SoftMaxWithCrossEntropy(n, net.input, net.ground_truth, loss_layer.delta, loss_layer.error);
      break;
  case CEW:
      int n = loss_layer.batch_size * loss_layer.input_size;
      CrossEntropy(n, net.input, net.ground_truth, loss_layer.delta, loss_layer.error);
      break;
  
  default:
      break;
  }

}