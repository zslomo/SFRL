#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sfrl/loader/loader.h"

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
