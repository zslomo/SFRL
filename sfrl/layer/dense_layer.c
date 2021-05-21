#include "sfrl/layer/dense_layer.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

DenseLayer MakeDenseLayer(int batch_size, int input_size, int output_size,
                          ActiType acti_type, InitType init_type) {
  DenseLayer layer = {0};
  layer.layer_type = DENSE;
  layer.batch_size = batch_size;
  layer.input_size = input_size;
  layer.output_size = output_size;
  layer.acti_type = acti_type;

  layer.output = calloc(input_size * batch_size, sizeof(float));
  // w, b, delta
  layer.delta = calloc(input_size * batch_size, sizeof(float));
  layer.weights = calloc(output_size * input_size, sizeof(float));
  layer.biases = calloc(output_size, sizeof(float));
  layer.weight_updates = calloc(input_size * output_size, sizeof(float));
  layer.bias_updates = calloc(output_size, sizeof(float));
  InitLayer(layer.weights, layer.biases, input_size, output_size, init_type);

  layer.forward = ForwardDenseLayer;
  layer.backward = BackwardDenseLayer;
  layer.update = UpdateDenseLayer;

  return layer;
}