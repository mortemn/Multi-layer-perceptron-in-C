#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "network.h"

struct Matrix {
    int rows;
    int cols;
    float **data;
};

// Samples from a normal distribution using Marsaglia polar method.
float randn() {
    static bool hasSpare = false;
    static float spare;
    if(hasSpare) {
        hasSpare = false;
        return spare;
    } else {
        hasSpare = true;
        float u, v, s;
        do {
            u = (rand() / ((float) RAND_MAX)) * 2.0 - 1.0;
            v = (rand() / ((float) RAND_MAX)) * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1 || s == 0);
        s = sqrt(-2.0 * log(s) / s);
        spare = v * s;
        return u * s;
    }
}

void init_matrix(struct Matrix *matrix, int rows, int cols) {
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        matrix->data[i] = malloc(cols * sizeof(float));
    }
}

void free_matrix(struct Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
}

void mul_matrix(struct Matrix *a, struct Matrix *b, struct Matrix *c) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            c->data[i][j] = 0;

            for (int k = 0; k < a->cols; k++) {
                c->data[i][j] += a->data[i][k] * b->data[k][j];
            }
    }
    c->rows = a->rows;
    c->cols = b->cols;
    }
}

void print_matrix(struct Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

void randn_matrix(struct Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] = randn();
        }
    }
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

void shuffle(int *array, size_t n) {
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}
