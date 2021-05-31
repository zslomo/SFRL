#include "sfrl/data/data.h"
#include <stdlib.h>

void FreeData(Data *data) {
  free(data->X);
  free(data->Y);
  free(data);
}