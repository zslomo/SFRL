#include "sfrl/blob/matrix.h"
#include "assert.h"
#include "sfrl/utils/blas.h"

Matrix MakeMatrix(int rows, int cols) {
  Matrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.vals = calloc(matrix.rows, sizeof(float *));
  for (int i = 0; i < matrix.rows; ++i) {
    matrix.vals[i] = calloc(matrix.cols, sizeof(float));
  }
  return matrix;
}

void FreeMatrix(Matrix matrix) {
  for (int i = 0; i < matrix.rows; ++i) {
    free(matrix.vals[i]);
  }
  free(matrix.vals);
}

Matrix CopyMatrix(Matrix m_source) {
  Matrix m_dist = {0};
  m_dist.cols = m_source.cols;
  m_dist.rows = m_source.rows;
  m_dist.vals = calloc(m_source.cols, sizeof(float *));
  for (int i = 0; i < m_source.rows; ++i) {
    m_source.vals[i] = calloc(m_source.cols, sizeof(float));
    for (int j = 0; j < m_source.cols; ++i) {
      m_dist.vals[i][j] = m_source.vals[i][j];
    }
  }
  return m_dist;
}

Matrix AddMatrix(Matrix m_source, Matrix m_dist) {
  assert(m_source.rows == m_source.rows);
  assert(m_source.cols == m_source.cols);
  for (int i = 0; i < m_source.rows; ++i) {
    for (int j = 0; j < m_source.cols; ++j) {
      m_dist.vals[i][j] += m_source.vals[i][j];
    }
  }
  return m_dist;
}

Matrix DotMatrix(Matrix m_source, Matrix m_dist) {
  assert(m_source.rows == m_source.rows);
  assert(m_source.cols == m_source.cols);
  for (int i = 0; i < m_source.rows; ++i) {
    for (int j = 0; j < m_source.cols; ++j) {
      m_dist.vals[i][j] *= m_source.vals[i][j];
    }
  }
  return m_dist;
}

void PrintMatrix(Matrix matrix) {
  printf("%d X %d Matrix:\n", matrix.rows, matrix.cols);
  printf(" __");
  for (int j = 0; j < 16 * matrix.cols - 1; ++j) {
    printf(" ");
  }

  printf("__ \n");

  printf("|  ");
  for (int j = 0; j < 16 * matrix.cols - 1; ++j) {
    printf(" ");
  }

  printf("  |\n");

  for (int i = 0; i < matrix.rows; ++i) {
    printf("|  ");
    for (int j = 0; j < matrix.cols; ++j) {
      printf("%15.7f ", matrix.vals[i][j]);
    }
    printf(" |\n");
  }
  printf("|__");
  for (int j = 0; j < 16 * matrix.cols - 1; ++j) {
    printf(" ");
  }

  printf("__|\n");
}
