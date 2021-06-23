#ifndef MERGE_LAYER_H
#define MERGE_LAYER_H
#include "base_layer.h"
#include "../network/network.h"
#include "../utils/init.h"

typedef Layer MergeLayer;

MergeLayer *MakeMergeLayer(int batch_size, int input_size, int output_size, int pre_layer_cnt,
                           int post_layer_cnt, MergeType merge_type, char *layer_name);
void ForwardMergeLayer(MergeLayer *layer);
void BackwardMergeLayer(MergeLayer *layer);

MergeSum(MergeLayer *layer);
MergeSumBackward(MergeLayer *layer);
MergeAvg(MergeLayer *layer);
MergeAvgBackward(MergeLayer *layer);
MergeDot(MergeLayer *layer);
MergeDotBackward(MergeLayer *layer);
MergeConcat(MergeLayer *layer);
MergeConcatBackward(MergeLayer *layer);

#endif