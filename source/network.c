#include <stdlib.h>
#include <stdio.h>
#include "math_utils.h"

extern int num_pixels;
extern int num_train;
extern int num_test;

struct Network {
    int num_layers;
    int *sizes;
    struct Matrix *biases;
    struct Matrix *weights;
};

void init_network(struct Network *network, int num_layers, int *sizes) {
    network->num_layers = num_layers;
    network->sizes = sizes;
    network->biases = malloc((num_layers - 1) * sizeof(struct Matrix));
    network->weights = malloc((num_layers - 1) * sizeof(struct Matrix));

    for (int i = 1; i < num_layers; i++) {
        init_matrix(&network->biases[i - 1], sizes[i], 1);
        init_matrix(&network->weights[i - 1], sizes[i], sizes[i - 1]);
    }

    for (int i = 0; i < num_layers - 1; i++) {
        randn_matrix(&network->biases[i]);
        randn_matrix(&network->weights[i]);
    }
}

// Forward propagation algorithm (feedforward).
void forward_prop(struct Network *network, struct Matrix *input, struct Matrix *output) {
    struct Matrix z;

    for (int i = 0; i < network->num_layers - 1; i++) {
        init_matrix(&z, network->weights[i].rows, input->cols);
        init_matrix(output, network->weights[i].rows, input->cols);

        // w * a + b.
        mul_matrix(&network->weights[i], input, &z);
        add_matrix(&z, &network->biases[i], output);
        sigmoid_matrix(output, output);

        free_matrix(input);
        *input = *output;
    }

    free_matrix(&z);
}

void sgd(struct Network *network, float data_train[num_pixels][num_train], int labels_train[num_train], float data_test[num_pixels][num_test], int labels_test[num_test], int epochs, int mini_batch_size, float eta) {

    shuffle(num_pixels, num_train, data_train);
    
    for (int i = 0; i < epochs; i++) {
    }
}
