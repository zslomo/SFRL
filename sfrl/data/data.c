#include "data.h"
#include "../utils/blas.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

Data *MakeData(int dims, int sample_size, int sample_num){
  Data *data = calloc(1, sizeof(Data));
  data->dims = dims;
  data->sample_size = sample_size;
  data->sample_num = sample_num;
  data->print_data = PrintData;
  data->normalize_data = NormalizeData;
  data->free_data = FreeData;

  return data;
}

void FreeData(Data *data) {
  free(data->X);
  free(data->Y);
  free(data);
}

void NormalizeData(Data *data) {
  int m = data->sample_size;
  int n = data->sample_num;
  float *mean = calloc(m, sizeof(float));
  float *variance = calloc(m, sizeof(float));
  MeanTensor(data->X, m, n, mean);
  VarianceTensor(data->X, m, n, mean, variance);
  NormTensor(data->X, m, n, mean, variance);
}

void PrintData(Data *data) {
  printf("size : %d \n", data->sample_num);
  printf("dims : %d \n", data->dims);
  printf("size_per_sample : %d \n", data->sample_size);
  int size = data->sample_size * data->sample_num;
  printf("X : ");
  for (int i = 0; i < size; ++i) {
    printf("%.2f ", data->X[i]);
  }
  printf("\n");
  printf("Y : ");
  for (int i = 0; i < data->sample_num; ++i) {
    printf("%.0f ", data->Y[i]);
  }
  printf("\n");
}