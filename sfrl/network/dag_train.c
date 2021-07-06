#include "dag_train.h"
#include <pthread.h>
#include "../type/type.h"

void *forward_thread(int thread_id, Layer *layer, Network *net, Layer *stop_layer) {
  // 处理一条线的情况，到merge 和 loss层结束
  while (layer->post_layer_cnt <= 1 && layer->layer_type == LOSS) {
    layer->ground_truth = net->ground_truth;
    layer->forward(layer, net);
    // 这里只处理一条线的情况， post只有一个
    layer = layer->post_layers[0];
  }
  stop_layer = layer;
  return 0;
}

void *backward_thread(int thread_id, Layer *layer, Network *net, Layer *stop_layer) {
  while (layer->pre_layer_cnt <= 1 && layer->layer_type == LOSS) {
    layer->backward(layer, net);
    layer = layer->pre_layers[0];
  }
  stop_layer = layer;
  return 0;
}

void *update_thread(int thread_id, Layer *layer, Network *net, Layer *stop_layer) {
  // 处理一条线的情况，到merge 和 loss层结束
  while (layer->post_layer_cnt <= 1 && layer->layer_type == LOSS) {
    layer->ground_truth = net->ground_truth;
    layer->update(layer, net);
    // 这里只处理一条线的情况， post只有一个
    layer = layer->post_layers[0];
  }
  stop_layer = layer;
  return 0;
}

void train_dag(Network *net) {
  pthread_t *threads = calloc(net->start_layer_cnt, sizeof(pthread_t *));
  Layer *layers = calloc(net->start_layer_cnt, sizeof(pthread_t *));
  for (int i = 0; i < net->start_layer_cnt; ++i) {
    int status = pthread_create(&threads[i], NULL, forward_thread, (void *)i, net->start_layers[i], net, layers[i]);
  }
}