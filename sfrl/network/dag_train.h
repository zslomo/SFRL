#ifndef DAG_TRAIN_H
#define DAG_TRAIN_H
#include "../layer/base_layer.h"
#include "network.h"

void *forward_thread(int thread_id, Layer *layer, Network *net, Layer *stop_layer);
void *backward_thread(int thread_id, Layer *layer, Network *net, Layer *stop_layer);
void *update_thread(int thread_id, Layer *layer, Network *net, Layer *stop_layer);
void train_dag(Network *net);

#endif