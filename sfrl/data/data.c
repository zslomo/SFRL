#include <stdio.h>
#include <stdlib.h>
#include "data.h"

void FreeData(Data *data) {
  free(data->X);
  free(data->Y);
  free(data);
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