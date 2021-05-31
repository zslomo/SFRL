#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #include "../sfrl/activation/activation.h"
// #include "../sfrl/data/data.h"
// #include "../sfrl/layer/base_layer.h"
// #include "../sfrl/layer/batchnorm_layer.h"
// #include "../sfrl/network/network.h"
// #include "../sfrl/optimizer/optimizer.h"
// #include "../sfrl/utils/blas.h"
#include "../sfrl/loader/loader.h"

float *ReadData(char *filename);


int main(int argc, char **argv) { ReadData("../../data/iris/iris.data"); }

float *ReadData(char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == 0) {
    printf("read file %s error", filename);
  }
  char *line;
  int nu = 0;
  while ((line = fgetl(file)) != 0) {
    strip(line);
    printf("%s\n", line);
  }
  fclose(file);
}