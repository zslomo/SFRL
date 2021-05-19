#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sfrl/activations/activations.h"
#include "sfrl/layer/activation_layer.h"
#include "sfrl/layer/base_layer.h"

Layer MakeActivationLayer(int batch_size, int inputs, ActiType acti_type) {
  Layer layer = {0};
  layer.type = ACTIVE;

  layer.inputs = inputs;
  layer.outputs = inputs;
  layer.batch_size = batch_size;

  layer.output = calloc(batch_size * inputs, sizeof(float*));
  layer.delta = calloc(batch_size * inputs, sizeof(float*));

  layer.forward = forward_activation_layer;
  layer.backward = backward_activation_layer;
  layer.activation = activation;
  fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
  return layer;
}

void forward_activation_layer(layer layer, network net) {
  copy_cpu(layer.outputs * layer.batch_size, net.input, 1, layer.output, 1);
  activate_array(layer.output, layer.outputs * layer.batch_size, layer.activation);
}

void backward_activation_layer(layer layer, network net) {
  gradient_array(layer.output, layer.outputs * layer.batch_size, layer.activation, layer.delta);
  copy_cpu(layer.outputs * layer.batch_size, layer.delta, 1, net.delta, 1);
}
