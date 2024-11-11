#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "network.h"
#include "math_utils.h"

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

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
    matrix->data = malloc(rows * cols * sizeof(float *));
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
    }
}

void scalar_mul_matrix(struct Matrix *a, float n, struct Matrix *b) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            b->data[i][j] = a->data[i][j] * n;
        }
    }
}

void hadamard_matrix(struct Matrix *a, struct Matrix *b, struct Matrix *c) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i][j] = a->data[i][j] * b->data[i][j];
        }
    }
}

void add_matrix(struct Matrix *a, struct Matrix *b, struct Matrix *c) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i][j] = a->data[i][j] + b->data[i][j]; 
        }
    }
}

void sub_matrix(struct Matrix *a, struct Matrix *b, struct Matrix *c) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i][j] = a->data[i][j] - b->data[i][j]; 
        }
    }
}

void sigmoid_matrix(struct Matrix *a, struct Matrix *b) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            b->data[i][j] = sigmoid(a->data[i][j]);
        }
    }
}

void sigmoid_derivative_matrix(struct Matrix *a, struct Matrix *b) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            b->data[i][j] = sigmoid_derivative(a->data[i][j]);
        }
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

void zero_matrix(struct Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] = 0;
        }
    }
}

void transpose_matrix(struct Matrix *a, struct Matrix *b) {
    init_matrix(b, a->cols, a->rows);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            b->data[j][i] = a->data[i][j];
        }
    }
}

void shuffle(int rows, int cols, float data[rows][cols]) {
    int rand_col = 0;
    float temp;

    for (int j = 0; j < cols; j++) {
        rand_col = rand() % cols;
            
        for (int i = 0; i < rows; i++) {
            temp = data[i][j];
            data[i][j] = data[i][rand_col];
            data[i][rand_col] = temp;
        }
    }
}
