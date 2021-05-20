#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "sfrl/layer/softmax_layer.h"

SoftmaxLayer MakeSoftmaxLayer(int batch_size, int input_size, int group_size) {
  assert(inputs % groups == 0);
  softmax_layer layer = {0};
  layer.layer_type = SOFTMAX;
  layer.batch_size = batch;
  layer.group_size = group_size;
  layer.input_size = input_size;  // softmax_layer的输入输出元素相同
  layer.output_size = input_size;

  layer.output = calloc(inputs * batch, sizeof(float));
  layer.delta = calloc(inputs * batch, sizeof(float));

  layer.forward = ForwardSoftmaxLayer;
  layer.backward = BackwardSoftmaxLayer;

  return layer;
}

void ForwardSoftmaxLayer(const SoftmaxLayer layer, NetWork net) {
  SoftmaxBatch(net.input, layer.input_size, layer.batch_size, layer.input_size,
               layer.temperature, layer.output);
}

void BackwardSoftmaxLayer(const SoftmaxLayer layer, NetWork net) {
  // 注意，这里的net.delta是 i+1层的 delta也就是 反向传播的上一层
  // 计算后赋值给当前层的delta layer.delta
  BackwardSoftmax(layer.output, layer.delta, layer.input_size, layer.batch_size,
                  layer.input_size, layer.temperature, net.delta);
}
