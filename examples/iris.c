#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../sfrl/activation/activation.h"
#include "../sfrl/data/data.h"
#include "../sfrl/layer/base_layer.h"
#include "../sfrl/layer/batchnorm_layer.h"
#include "../sfrl/layer/dense_layer.h"
#include "../sfrl/layer/loss_layer.h"
#include "../sfrl/layer/softmax_layer.h"
#include "../sfrl/loader/loader.h"
#include "../sfrl/network/network.h"
#include "../sfrl/optimizer/optimizer.h"
#include "../sfrl/utils/blas.h"

Data BuildInput(char **samples, int batch_size, int sample_num, int sample_size) {
  Data data = MakeData(2, sample_size, sample_num);
  data.X = calloc(sample_size * data.sample_num, sizeof(float));
  data.Y = calloc(sample_num, sizeof(float));
  // printf("size = %d \n", data.sample_num);
  // printf("sample : %s \n", samples[9]);
  for (int i = 0; i < sample_num; ++i) {
    // printf("line %d: %s\n", i, samples[i]);
    char **tokens = StrSplit(samples[i], ",");
    for (int j = 0; j < sample_size; ++j) {
      data.X[i * sample_size + j] = atof(tokens[j]);
    }
    if (strcmp(tokens[sample_size], "Iris-setosa")) {
      data.Y[i] = 0;
    } else {
      data.Y[i] = 1;
    }
  }
  // data.normalize_data(&data);
  int batch_num = data.sample_num / batch_size;
  int last_batch_size = data.sample_num % batch_size;
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
  int batch_size = 4;
  DenseLayer dnn_1 = MakeDenseLayer(batch_size, data->sample_size, 4, TANH, NORMAL, "dense_1");
  // DenseLayer dnn_2 = MakeDenseLayer(batch_size, 2, 2, TANH, NORMAL, "dense_2");
  SoftmaxLayer sm_1 = MakeSoftmaxLayer(batch_size, 2, "softmax");
  LossLayer loss_layer = MakeLossLayer(batch_size, 2, 2, CE, "loss");

  net->batch_size = batch_size;
  net->sample_size = data->sample_size;
  net->layers[0] = &dnn_1;
  // net->layers[1] = &dnn_2;
  net->layers[1] = &sm_1;
  net->layers[2] = &loss_layer;
  printf("start train...\n");
  net->learning_rate = 0.1;
  net->train(net, data, SGD, 10);
}
int main(int argc, char **argv) {
  printf("Read data...\n");
  char **samples = malloc(255 * sizeof(char *));
  int sample_num = ReadData("../data/iris/iris.data", samples);
  printf("get sample done...\n");
  Data data = BuildInput(samples, 16, sample_num, 4);
  Network net = MakeNetwork(3);
  BuildNet(&data, &net);
}