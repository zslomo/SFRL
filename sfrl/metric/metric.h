#ifndef METRIC_H
#define METRIC_H

typedef enum { MSE, ACC } MetricType;
/**
 *  说实话 对于强化学习来说，metric作用不大
 *  主要还是看reward，这里搞两个聊胜于无吧
 * */
float MseMetric(int n, float *pred, float *truth);
float AccMetric(int n, float threshold, float *pred, float *truth);

#endif