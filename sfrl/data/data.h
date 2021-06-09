#ifndef DATA_H
#define DATA_H

/**
 *  Data中存储的是已经结构化的数据
 *  为了最大程度的通用性，并且又不损失任何性能，那一切都按指针来
 *  不管二维三维，在c语言中都是按照行flat开拼在一起存储，无论是
 *  几维，本质上都是在指针指向的内存位置开辟了一块儿连续的空间，
 *  所以这里类似于数据的序列化，只要存好定义数据的元素，直接反序列化就可以拿到数据
 **/
typedef struct Data {
  float *X;
  float *Y;
  // 输入的维度
  // 维度一般有几种
  // 1 普通机器学习 2维 (batch_size, features)
  // 2 rnn        3维 (batch_size, time_step, features)
  // 3 cnn        4维 (batch_size, width, height, channel)
  int dims;
  int sample_num;
  // 每个样本的数据占多大
  // 其实就是看有几个维度就吧对应的dn连成起来
  int sample_size;
  void (*free_data)(struct Data *);
  void (*print_data)(struct Data *);
  void (*normalize_data)(struct Data *);
} Data;

Data MakeData(int dims, int sample_size, int sample_num);
void FreeData(Data *data);
void PrintData(Data *data);
void NormalizeData(Data *data);

#endif