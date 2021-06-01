#include "loader.h"
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 *  按行读文件的代码直接copy了darnet的函数
 *  写的很好，我就不自己写垃圾代码啦
 * */
char *FileGetLine(FILE *fp) {
  if (feof(fp)) {
    return 0;
  }

  size_t size = 512;
  char *line = malloc(size * sizeof(char));
  if (!fgets(line, size, fp)) {
    free(line);
    return 0;
  }
  size_t curr = strlen(line);
  while ((line[curr - 1] != '\n') && !feof(fp)) {

    if (curr == size - 1) {
      size *= 2;
      line = realloc(line, size * sizeof(char));
      if (!line) {
        printf("%ld\n", size);
        printf("malloc fail!\n");
      }
    }

    size_t readsize = size - curr;
    if (readsize > INT_MAX) {
      readsize = INT_MAX - 1;
    }

    fgets(&line[curr], readsize, fp);
    curr = strlen(line);
  }
  if (line[curr - 1] == '\n') {
    line[curr - 1] = '\0';
  }

  return line;
}

void Strip(char *s) {
  size_t i;
  size_t len = strlen(s);
  size_t offset = 0;
  for (i = 0; i < len; ++i) {
    char c = s[i];
    if (c == ' ' || c == '\t' || c == '\n')
      ++offset;
    else
      s[i - offset] = c;
  }
  s[len - offset] = '\0';
}

/**
 *  分割字符串
 *  c 的字符串分割是真的沙雕,strtok接受 char*会炸
 *  只能自己写，
 **/

char **StrSplit(char *str, char *delim) {
  int origin_str_size = -1;
  int delim_size = -1;
  // 遍历得到原字符串和 分割串的长度
  while (str[++origin_str_size] != '\0')
    ;
  while (delim[++delim_size] != '\0')
    ;
  // 如果原串最后不是 分割串， 那么加上一个分割串
  // 这样处理的原因是逻辑是遇到每个分割串时， 分割前面的子串，
  // 不加的话最后一个子串不好处理，这样方便处理
  // 但是要记住，这个操作是会改变原值的，后面要记得改回来
  if(!StringCmp(str + origin_str_size - delim_size, delim, delim_size)){
    str = realloc(str, (origin_str_size + delim_size + 1) * sizeof(char));
    for(int i = 0; i < delim_size; ++i){
      str[origin_str_size + i] = delim[i];
    }
  }
  str[origin_str_size + delim_size] = '\0';
  int str_size = origin_str_size + delim_size + 1;

  int init_size = 255;
  char **result = malloc(init_size * sizeof(char *));
  int res_index = 0;
  // i j 双指针，j指向待处理的子串头，i寻找分割串，找到后 j ~ i 之间就是待分割子串
  int j = 0;
//   printf("delim_size = %d, str_size = %d, res_index = %d\n", delim_size, str_size, res_index);
  for (int i = 0; i < str_size; ++i) {
    if ((i < str_size - delim_size && StringCmp(str + i, delim, delim_size))) {   
      if (res_index == init_size) {
        init_size *= 2;
        result = realloc(result, init_size * sizeof(char *));
      }
      result[res_index] = malloc((i - j + 1) * sizeof(char));
      memcpy(result[res_index], str + j, (i - j) * sizeof(char));
      result[res_index][i - j] = '\0';
      i += delim_size;
      j = i;
    //   printf("result[res_index] = %s, res_index = %d, res_index = %d\n", result[res_index],
            //  res_index, res_index);
    //   printf("i = %d, j= %d \n", i, j);
      res_index++;
    }
  }
  str = realloc(str, origin_str_size*sizeof(char));
  str[origin_str_size - 1] = '\0';
//   printf("str = \"%s\"\n", str);
  result = realloc(result, res_index * sizeof(char *));
  for (int i = 0; i < res_index; ++i){
      printf("str = \"%s\"\n", result[i]);
  }
    return result;
}

int StringCmp(const char *src, const char *dst, int size) {
  for (int i = 0; i < size; i++) {
    if (dst[i] == '\0' || src[i] == '\0' || dst[i] != src[i]) {
      return 0;
    }
  }
  return 1;
}