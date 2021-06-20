#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "../layer/base_layer.h"
#include "../network/network.h"
/**
 * 优化方法
 * 这里讲的非常详细 https://d2l.ai/chapter_optimization/sgd.html
 **/

void SgdOptimizer(Network *net, Layer *layer);
void AdaGradOptimizer(Network *net, Layer *layer);
void RmsPropOptimizer(Network *net, Layer *layer);
void AdamOptimizer(Network *net, Layer *layer);

#endif