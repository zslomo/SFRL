#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../sfrl/activation/activation.h"
#include "../sfrl/data/data.h"
#include "../sfrl/layer/base_layer.h"
#include "../sfrl/layer/batchnorm_layer.h"
#include "../sfrl/layer/dense_layer.h"
#include "../sfrl/layer/loss_layer.h"
#include "../sfrl/layer/merge_layer.h"
#include "../sfrl/layer/softmax_layer.h"
#include "../sfrl/loader/loader.h"
#include "../sfrl/network/network.h"
#include "../sfrl/optimizer/optimizer.h"
#include "../sfrl/utils/blas.h"

Data *BuildInput(char **samples, int batch_size, int sample_num, int sample_size) {
  Data *data = MakeData(2, sample_size, sample_num);
  data->X = calloc(sample_size * data->sample_num, sizeof(float));
  data->Y = calloc(sample_num, sizeof(float));
  // printf("size = %d \n", data->sample_num);
  // printf("sample : %s \n", samples[9]);
  for (int i = 0; i < sample_num; ++i) {
    // printf("line %d: %s\n", i, samples[i]);
    char **tokens = StrSplit(samples[i], ",");
    for (int j = 0; j < sample_size; ++j) {
      data->X[i * sample_size + j] = atof(tokens[j]);
    }
    if (!strcmp(tokens[sample_size], "Iris-setosa")) {
      data->Y[i] = 0;
    } else if (!strcmp(tokens[sample_size], "Iris-versicolor")) {
      data->Y[i] = 1;
    } else {
      data->Y[i] = 2;
    }
  }
  data->class_num = 3;
  // data->normalize_data(data);
  int batch_num = data->sample_num / batch_size;
  int last_batch_size = data->sample_num % batch_size;
  // PrintData(&data);
  return data;
}

int ReadData(char *filename, char **samples) {
  FILE *file = fopen(filename, "r");
  if (file == 0) {
    printf("read file %s error", filename);
  }
  char *line;
  int init_size = 255;
  int sample_num = -1;
  while ((line = FileGetLine(file)) != 0) {
    Strip(line);
    samples[++sample_num] = line;
    if (sample_num + 1 == init_size) {
      init_size *= 2;
      samples = realloc(samples, init_size * sizeof(char *));
    }
  }
  fclose(file);
  return sample_num + 1;
}

int BuildNet(Data *data, Network *net) {
  int class_num = 3;
  int seed = 1024;

  // 构建节点
  Layer *dense_1 =
      MakeDenseLayer(net->batch_size, data->sample_size, 16, 0, 1, LINEAR, NORMAL, seed, "dense_1");
  Layer *dense_2 =
      MakeDenseLayer(net->batch_size, data->sample_size, 16, 0, 1, LINEAR, NORMAL, seed, "dense_2");
  Layer *dense_3 =
      MakeDenseLayer(net->batch_size, data->sample_size, 16, 0, 1, LINEAR, NORMAL, seed, "dense_3");
  Layer *merge_1 = MakeMergeLayer(net->batch_size, 16, 16, 3, 1, AVG, "merge_1");
  Layer *dense_4 = MakeDenseLayer(net->batch_size, 16, 8, 1, 1, LINEAR, NORMAL, seed, "dense_4");
  Layer *dense_5 =
      MakeDenseLayer(net->batch_size, data->sample_size, 16, 0, 1, LINEAR, NORMAL, seed, "dense_5");
  Layer *dense_6 = MakeDenseLayer(net->batch_size, 16, 8, 1, 1, LINEAR, NORMAL, seed, "dense_6");
  Layer *merge_2 = MakeMergeLayer(net->batch_size, 8, 8, 2, 1, SUM, "merge_2");
  Layer *dense_7 = MakeDenseLayer(net->batch_size, 8, 4, 1, 2, LINEAR, NORMAL, seed, "dense_7");
  Layer *dense_8 =
      MakeDenseLayer(net->batch_size, 4, class_num, 1, 1, LINEAR, NORMAL, seed, "dense_8");
  Layer *dense_9 =
      MakeDenseLayer(net->batch_size, 4, class_num, 1, 1, LINEAR, NORMAL, seed, "dense_9");
  Layer *sm_1 = MakeSoftmaxLayer(net->batch_size, class_num, 1, 1, "softmax_1");
  Layer *sm_2 = MakeSoftmaxLayer(net->batch_size, class_num, 1, 1, "softmax_2");
  Layer *loss_1 = MakeLossLayer(net->batch_size, class_num, class_num, CE, "loss_1");
  Layer *loss_2 = MakeLossLayer(net->batch_size, class_num, class_num, CE, "loss_2");

  // 画计算图 
  LinkLayers(dense_1, merge_1);
  LinkLayers(dense_2, merge_1);
  LinkLayers(dense_3, merge_1);
  LinkLayers(merge_1, dense_4);
  LinkLayers(dense_5, dense_6);
  LinkLayers(dense_4, merge_2);
  LinkLayers(dense_6, merge_2);
  LinkLayers(merge_2, dense_7);
  LinkLayers(dense_7, dense_8);
  LinkLayers(dense_7, dense_9);
  LinkLayers(merge_2, dense_7);
  LinkLayers(dense_8, sm_1);
  LinkLayers(sm_1, loss_1);
  LinkLayers(dense_9, sm_2);
  LinkLayers(sm_2, loss_2);

  Layer start_layer_list[4] = {dense_1, dense_2, dense_3, dense_5};
  net->start_layer_cnt = 4;
  net->start_layers = start_layer_list;
  Layer loss_layer_list[2] = {dense_1, dense_2, dense_3, dense_5};
  net->loss_layer_cnt = 2;
  net->loss_layers = loss_layer_list;

  net->sample_size = data->sample_size;
  net->pred = calloc(net->batch_size * class_num, sizeof(float));
}

int main(int argc, char **argv) {
  clock_t b_time = clock();
  printf("Read data...\n");
  char **samples = malloc(255 * sizeof(char *));
  int sample_num = ReadData("../data/iris/iris.data", samples);
  int batch_size = 150;
  printf("get sample done...\n");
  Data *data = BuildInput(samples, batch_size, sample_num, 4);
  Network *net = MakeNetwork(5, batch_size);
  printf("make network done..\n");
  BuildNet(data, net);
  printf("start train...\n");
  net->learning_rate = 0.1;
  
  net->train(net, data, SGD, 10);
  printf("time cost %f\n", (clock() - b_time) * 1.0 / CLOCKS_PER_SEC);
  net->test(net, data);
}