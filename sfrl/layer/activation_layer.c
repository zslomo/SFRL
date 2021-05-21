#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sfrl/activations/activations.h"
#include "sfrl/layer/activation_layer.h"
#include "sfrl/layer/base_layer.h"

Layer MakeActivationLayer(int batch_size, int input_size, ActiType acti_type) {
  Layer layer = {0};
  layer.acti_type = ACTIVE;

  layer.input_size = input_size;
  layer.output_size = input_size;
  layer.batch_size = batch_size;

  layer.output = calloc(batch_size * inputs, sizeof(float *));
  layer.delta = calloc(batch_size * inputs, sizeof(float *));

  layer.forward = ForwardActivationLayer;
  layer.backward = BackwardActivationLayer;
  return layer;
}

void ForwardActivationLayer(ActivationLayer layer, NetWork net) {
  copy_cpu(layer.outputs * layer.batch_size, net.input, 1, layer.output, 1);
  ActivateTensor(layer.output, layer.output_size * layer.batch_size, layer.activation);
}

void BackwardActivationLayer(ActivationLayer layer, NetWork net) {
  gradient_array(layer.output, layer.outputs * layer.batch_size, layer.activation, layer.delta);
  copy_cpu(layer.outputs * layer.batch_size, layer.delta, 1, net.delta, 1);
}


