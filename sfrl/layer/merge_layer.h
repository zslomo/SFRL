#ifndef MERGE_LAYER_H
#define MERGE_LAYER_H
#include "../network/network.h"
#include "../utils/init.h"
#include "base_layer.h"

typedef Layer MergeLayer;

MergeLayer *MakeSoftmaxLayer(int batch_size, int input_size, int pre_layer_cnt, int post_layer_cnt, char *layer_name);
void ForwardMergeLayer(MergeLayer *layer, Network *net);
void BackwardMergeLayer(MergeLayer *layer, Network *net);

void MergeSum(MergeLayer *layer);
void MergeSumBackward(MergeLayer *layer);
void MergeAvg(MergeLayer *layer);
void MergeAvgBackward(MergeLayer *layer);
void MergeDot(MergeLayer *layer);
void MergeDotBackward(MergeLayer *layer);
void MergeConcat(MergeLayer *layer);
void MergeConcatBackward(MergeLayer *layer);

#endif