#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "activation.h"

static inline float SigmoidActivate(float x) { return 1. / (1. + exp(-x)); }
static inline float SigmoidGradient(float x) { return (1 - x) * x; }

static inline float ReluActivate(float x) { return x * (x > 0); }
static inline float ReluGradient(float x) { return (x > 0); }

static inline float TanhActivate(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }
static inline float TanhGradient(float x) { return 1 - x * x; }

float Activate(float x, ActiType acti_type) {
  switch (acti_type) {
  case LINEAR:
    return x;
  case SIGMOID:
    return SigmoidActivate(x);
  case RELU:
    return ReluActivate(x);
  case TANH:
    return TanhActivate(x);
  }
  return 0;
}

float Gradient(float x, ActiType acti_type) {
  switch (acti_type) {
  case LINEAR:
    return 1;
  case SIGMOID:
    return SigmoidGradient(x);
  case RELU:
    return ReluGradient(x);
  case TANH:
    return TanhGradient(x);
  }
  return 0;
}

void ActivateTensor(float *TensorX, const int size, const ActiType acti_type) {

  for (int i = 0; i < size; ++i) {
    TensorX[i] = Activate(TensorX[i], acti_type);
  }
}

void GradientTensor(const float *TensorX, const int size, const ActiType acti_type, float *delta) {
  for (int i = 0; i < size; ++i) {
    delta[i] *= Gradient(TensorX[i], acti_type);
  }
}
