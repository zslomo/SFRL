#ifndef LOADER_H
#define LOADER_H
#include <stdio.h>

void Strip(char *s);
char *FileGetLine(FILE *fp);
char **StrSplit(char *str, char *delim);
char *strdup(const char *src);

#endif