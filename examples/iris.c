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
#include "../sfrl/loader/loader.h"
#include "../sfrl/network/network.h"
#include "../sfrl/optimizer/optimizer.h"
#include "../sfrl/utils/blas.h"

void ReadData(char *filename);

int main(int argc, char **argv) {
  printf("Read data...\n");
  ReadData("../data/iris/iris.data");
}

void ReadData(char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == 0) {
    printf("read file %s error", filename);
  }
  char *line;
  Data data = {0};
  data.dims = 2;
  data.size_per_sample = 4;
  int init_size = 255;  
  int sample_size = -1;
  char **samples = malloc(init_size * sizeof(char *));
  while ((line = FileGetLine(file)) != 0) {
    Strip(line);
    samples[++sample_size] = line;
    if (sample_size == init_size) {
      init_size *= 2;
      samples = realloc(samples, init_size * sizeof(char *));
    }
  }
  data.size = sample_size + 1;
  samples = realloc(samples, data.size * sizeof(char *));
  data.X = calloc(data.size_per_sample * data.size, sizeof(float));
  data.Y = calloc(data.size, sizeof(float));

  printf("size = %d \n", data.size);
  // printf("sample : %s \n", samples[9]);
  for (int i = 0; i < data.size; ++i) {
    printf("line %d: %s\n", i, samples[i]);
    char **tokens = StrSplit(samples[i], ",");
    for (int j = 0; j < 4; ++j) {
      data.X[i * 4 + j] = atof(tokens[j]);
    }
    if (strcmp(tokens[4], "Iris-setosa")) {
      data.Y[i] = 1;
    } else {
      data.Y[i] = 2;
    }
  }
  fclose(file);
  int batch_num = data.size / 16;
  int last_batch_size = data.size % 16;

  PrintData(&data);
}