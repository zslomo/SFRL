#include "utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int GetStringCharCount(char const *str) {
  int size = -1;
  while (str[++size] != '\0')
    ;
  return size;
}

int GetIntCharCount(int num) {
  if (num = 0) {
    return 0;
  }
  int size = 0;
  while (num != 0) {
    num /= 10;
    size++;
  }
  return size;
}

void PrintGridOutline(int size) {
  printf("|");
  for (int i = 0; i < size; ++i) {
    printf("-");
  }
  printf("|\n");
}

void PrintGridColums(int size, int num) {
  printf("|  ");
  for (int j = 0; j < num - 2; ++j) {
    printf(" ");
  }
  for (int i = 0; i < size; ++i) {
    printf("| %d", i);
    int size_pre_cell = num - GetIntCharCount(i);
    for (int j = 0; j < size_pre_cell - 2; ++j) {
      printf(" ");
    }
  }
  printf("|\n");
}

void PrintGridInnerline(int size, int num) {
  for (int i = 0; i < size; ++i) {
    printf("|");
    for (int j = 0; j < num; ++j) {
      printf("-");
    }
  }
  printf("|\n");
}

char *FloatToString(int size, float num) {
  int len = snprintf(NULL, 0, "%f", num);
  if (size == 0) {
    size = len;
  }

  if (pow(10, size + 1) - 1 < num) {
    char *s = calloc(size + 1, sizeof(char));
    memset(s, 'A', size);
    s[size] = '\0';
    return s;
  }

  if (-pow(10, size + 1) + 1 > num) {
    char *s = calloc(size + 2, sizeof(char));
    s[0] = ' ';
    memset(s + 1, 'A', size);
    s[size + 1] = '\0';
    return s;
  }
  if (num < 0) {
    size++;
  }

  char *ret = malloc((len + 1) * sizeof(char));
  snprintf(ret, (len + 1), "%f", num);
  if (ret[size - 1] == '.') {
    printf("float print size too small... please puls 1\n");
    assert(ret[size - 1] != '.');
  }
  ret = realloc(ret, (size + 1) * sizeof(char));
  ret[size] = '\0';
  return ret;
}

void PrintTensor2D(float *Tensor, int n, int m) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      printf("%f ", Tensor[i * m + j]);
    }
    printf("\n");
  }
}

char *GetOptimizerStr(OptType opt_type) {
  char *opt_str;
  if (opt_type == ADAM) {
    opt_str = "adam";
  } else if (opt_type == SGD) {
    opt_str = "sgd";
  } else if (opt_type == ADAGRAD) {
    opt_str = "adagrad";
  } else if (opt_type == RMSPROP) {
    opt_str = "rmsprop";
  } else {
    opt_str = "error";
  }
  return opt_str;
}

char *GetLayerTypeStr(LayerType layer_type) {
  char *layer_type_str;
  if (layer_type == DENSE) {
    layer_type_str = "Dense";
  } else if (layer_type == BATCHNORMALIZATION) {
    layer_type_str = "BatchNorm";
  } else if (layer_type == SOFTMAX) {
    layer_type_str = "SoftMax";
  } else if (layer_type == DROPOUT) {
    layer_type_str = "DropOut";
  } else if (layer_type == ACTIVATION) {
    layer_type_str = "Activation";
  } else if (layer_type == LOSS) {
    layer_type_str = "Loss";
  } else {
    layer_type_str = "error";
  }
  return layer_type_str;
}