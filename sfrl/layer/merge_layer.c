#include "../utils/blas.h"
#include "../utils/utils.h"
#include "contract_layer.h"

MergeLayer *MakeMergeLayer(int batch_size, int input_size, int output_size, MergeType merge_type,
                           char *layer_name) {
  MergeLayer *layer = calloc(1, sizeof(MergeLayer));
  layer->layer_type = MERGE;
  layer->merge_type = merge_type;
  layer->layer_name = layer_name;
  layer->batch_size = batch_size;
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->delta = calloc(batch_size * output_size, sizeof(float));
  layer->output = calloc(output_size * batch_size, sizeof(float));
  layer->input = calloc(input_size * batch_size, sizeof(float));

  layer->forward = ForwardMergeLayer;
  layer->backward = BackwardMergeLayer;

  return layer;
}

void ForwardMergeLayer(MergeLayer *layer) {
  assert(layer->pre_layer_cnt > 0);
  assert(layer->pre_layers);
  assert(layer->post_layer_cnt > 0);
  assert(layer->post_layers);
  switch (layer->merge_type) {
  case SUM:
    MergeSum(layer);
    break;
  case AVG:
    MergeAvg(layer);
    break;
  case DOT:
    MergeDot(layer);
    break;
  default:
    break;
  }
}

void BackwardMergeLayer(MergeLayer *layer) {
  assert(layer->pre_layer_cnt > 0);
  assert(layer->pre_layers);
  assert(layer->post_layer_cnt > 0);
  assert(layer->post_layers);
  switch (layer->merge_type) {
  case SUM:
    MergeSumBackward(layer);
    break;
  case AVG:
    MergeAvgBackward(layer);
    break;
  case DOT:
    MergeDotBackward(layer);
    break;
  default:
    break;
  }
}

MergeSum(MergeLayer *layer) {
  for (int i = 0; i < layer->pre_layer_cnt; ++i) {
    Layer *pre_layer = layer->pre_layers[i];
    assert(layer->output_size == pre_layer->output_size);
    AxpyTensor(layer->output_size, 1, pre_layer->output, layer->output);
  }
}

/**
 *  sum模式下，merge层的输出是所有输入的和
 *      out = input_1 + input_2 +...+input_n
 *  输入又是pre层的输出
 *       z    = w*input_pre + bias_pre
 *      input = Activate(z)
 *  输出是所有输入的和，导数其实也一样，导数传递时每个w得到的其实只有自己提供的那一部分
 *      dout/dw_1 = dout/dinput_1 * dinput_1/dw_1 + dout/dinput_2 * dinput_2/dw_1 + ...
 *  跟自己无关的input 对w求导都是0，
 *      dout/dw_1 = dout/dinput_1 * dinput_1/dz_1 * dz_1 /dw_1
 *                = delta * dacti * input_pre_1
 *  So sum模式下的 梯度回传就是delta透穿给每个子层而已
 **/

MergeSumBackward(MergeLayer *layer) {
  int layer_cnt = layer->pre_layer_cnt;
  for (int i = 0; i < layer->pre_layer_cnt; ++i) {
    Layer *pre_layer = layer->pre_layers[i];
    assert(layer->output_size == pre_layer->output_size);
    memcpy(pre_layer->delta, layer->delta, layer->output_size * sizeof(float));
  }
}

MergeAvg(MergeLayer *layer) {
  int layer_cnt = layer->pre_layer_cnt;
  for (int i = 0; i < layer_cnt; ++i) {
    Layer *pre_layer = layer->pre_layers[i];
    assert(layer->output_size == pre_layer->output_size);
    AxpyTensor(layer->output_size, 1.0 / layer_cnt, pre_layer->output, layer->output);
  }
}

/**
 *  avg模式下，merge层的输出是所有输入的平均值
 *      out = (input_1 + input_2 +...+input_n) / n
 *  输入又是pre层的输出
 *       z    = w*input_pre + bias_pre
 *      input = Activate(z)
 *  输出是所有输入的和，导数其实也一样，导数传递时每个w得到的其实只有自己提供的那一部分
 *      dout/dw_1 = dout/dinput_1 * dinput_1/dw_1 + dout/dinput_2 * dinput_2/dw_1 + ...
 *  跟自己无关的input 对w求导都是0，
 *      dout/dw_1 = dout/dinput_1 * dinput_1/dz_1 * dz_1 /dw_1
 *                = delta / n * dacti * input_pre_1
 *  So avg模式下的 梯度回传就是 delta / n
 **/

MergeAvgBackward(MergeLayer *layer) {
  int layer_cnt = layer->pre_layer_cnt;
  for (int i = 0; i < layer->pre_layer_cnt; ++i) {
    Layer *pre_layer = layer->pre_layers[i];
    assert(layer->output_size == pre_layer->output_size);
    AxpyTensor(layer->output_size, 1.0 / layer_cnt, layer->delta, pre_layer->delta);
  }
}

MergeDot(MergeLayer *layer) {
  int layer_cnt = layer->pre_layer_cnt;
  InitTensor(layer->output_size, 1.0, layer->output);
  for (int i = 0; i < layer_cnt; ++i) {
    Layer *pre_layer = layer->pre_layers[i];
    assert(layer->output_size == pre_layer->output_size);
    DotTensor(layer->output_size, pre_layer->output, layer->output);
  }
}

/**
 *  dot模式下，merge层的输出是所有输入的点乘
 *      out = input_1 *input_2 *...*input_n
 *  输入又是pre层的输出
 *       z    = w*input_pre + bias_pre
 *      input = Activate(z)
 *  输出是所有输入的和，导数其实也一样，导数传递时每个w得到的除了自己提供的部分，还有所有其他输入
 *  但是同样 除了dinput_1/dw_1 外，其他都是0，都可以忽略
 *      dout/dw_1 = dout/dinput_1 * dinput_1/dw_1 + dout/dinput_2 * dinput_2/dw_1 + ...
 *                = dout/dinput_1 * dinput_1/dz_1 * dz_1 /dw_1
 *                = delta *(input_2 *...*input_n) * dacti * input_pre_1
 *  So dot模式下的 梯度回传就是 delta * (input_2 *...*input_n)
 **/

MergeDotBackward(MergeLayer *layer) {
  int layer_cnt = layer->pre_layer_cnt;
  float *tmp = malloc(layer->output_size * sizeof(float));
  InitTensor(layer->output_size, 1.0, tmp);
  for (int i = 0; i < layer_cnt; ++i) {
    Layer *pre_layer = layer->pre_layers[i];
    assert(layer->output_size == pre_layer->output_size);
    InitTensor(layer->output_size, 1.0, tmp);
    for (int j = 0; j < layer_cnt; ++j) {
      if (i != j) {
        DotTensor(layer->output_size, pre_layer->output, layer->tmp);
      }
    }
    DotTensor(layer->output_size, layer->tmp, pre_layer->delta);
  }
  free(tmp);
}