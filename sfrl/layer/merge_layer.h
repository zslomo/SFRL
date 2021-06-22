#ifndef MERGE_LAYER_H
#define MERGE_LAYER_H
#include "base_layer.h"
#include "../network/network.h"
#include "../utils/init.h"

typedef Layer MergeLayer;

MergeLayer *MakeMergeLayer(int batch_size, int input_size, int output_size, MergeType merge_type, char *layer_name);
void ForwardMergeLayer(MergeLayer *layer, Network *net);
void BackwardMergeLayer(MergeLayer *layer, Network *net);

MergeSum(MergeLayer *layer);
MergeSumBackward(MergeLayer *layer);
MergeAvg(MergeLayer *layer);
MergeAvgBackward(MergeLayer *layer);
MergeDot(MergeLayer *layer);
MergeDotBackward(MergeLayer *layer);
#endif