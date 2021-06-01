#include "data.h"
#include <stdlib.h>

void FreeData(Data *data) {
  free(data->X);
  free(data->Y);
  free(data);
}

void PrintData(Data *data) {
  printf("size : %d \n", data->size);
  printf("dims : %d \n", data->dims);
  printf("size_per_sample : %d \n", data->size_per_sample);
  int size = data->size_per_sample * data->size;
  printf("X : ");
  for (int i = 0; i < size; ++i) {
    printf("%.2f ", data->X[i]);
  }
  printf("\n");
  printf("Y : ");
  for (int i = 0; i < data->size; ++i) {
    printf("%.0f ", data->Y[i]);
  }
  printf("\n");
}