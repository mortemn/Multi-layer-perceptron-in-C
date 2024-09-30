#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stdlib.h>

typedef struct Matrix {
	int rows;
	int cols;
	float **data;
} Matrix;

void init_matrix(Matrix *matrix, int rows, int columns);

void randn_matrix(Matrix *matrix);

void free_matrix(Matrix *matrix);

void mul_matrix(Matrix *a, Matrix *b, Matrix *c);

void add_matrix(Matrix *a, Matrix *b, Matrix *c);

void sigmoid_matrix(Matrix *a, Matrix *b);

void print_matrix(Matrix *matrix);

float sigmoid(float);

float sigmoid_derivative(float);

float randn();

void shuffle(int *, size_t);

#endif
