#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include <math.h>
#include "../type/type.h"
// 激活函数
float Activate(float x, ActiType acti_type);
float Gradient(float x, ActiType acti_type);
void ActivateTensor(float *TensorX, const int size, const ActiType acti_type);
void GradientTensor(const float *TensorX, const int size, const ActiType acti_type, float *delta);

static float SigmoidActivate(float x);
static float SigmoidGradient(float x);

static float ReluActivate(float x);
static float ReluGradient(float x);

static float TanhActivate(float x);
static float TanhGradient(float x);

char *GetActivationTypeStr(ActiType acti_type);

#endif
