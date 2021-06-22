#ifndef LOADER_H
#define LOADER_H
#include <stdio.h>
#include "../loss/loss.h"
#include "../type/type.h"

void Strip(char *s);
char *FileGetLine(FILE *fp);
char **StrSplit(char *str, char *delim);
char *strdup(const char *src);
char *GetLossStr(LossType loss_type);
int StringCmp(const char *src, const char *dst, int size);

#endif