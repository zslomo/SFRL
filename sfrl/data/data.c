#include <stdlib.h>
#include "data.h"

void FreeData(Data *data) {
  free(data->X);
  free(data->Y);
  free(data);
}