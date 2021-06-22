#ifndef UTILS_H
#define UTILS_H
#include "../layer/base_layer.h"
#include "../optimizer/optimizer.h"

int GetStringCharCount(char const *str);
int GetIntCharCount(int num);
void PrintGridOutline(int size);
void PrintGridInnerline(int size, int num);
char *FloatToString(int size, float num);
void PrintGridColums(int size, int num);
void PrintTensor2D(float *Tensor, int n, int m);
char *GetOptimizerStr(OptType opt_type);
char *GetLayerTypeStr(LayerType layer_type);

#endif