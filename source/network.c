#include <stdlib.h>
#include <stdio.h>
#include "math_utils.h"
#include "network.h"

extern int num_pixels;
extern int num_train;
extern int num_test;

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

void index_input(float data[num_pixels][num_train], int index, struct Matrix *input) {
    init_matrix(input, num_pixels, 1);
    for (int i = 0; i < num_pixels; i++) {
        input->data[i][0] = data[i][index];
    }
}

Updates backprop(struct Network *network, Matrix *input, int label) {
    Updates updates;
    updates.nabla_w = malloc((network->num_layers - 1) * sizeof(struct Matrix));
    updates.nabla_b = malloc((network->num_layers - 1) * sizeof(struct Matrix));

    for (int i = 0; i < network->num_layers - 1; i++) {
        init_matrix(&updates.nabla_w[i], network->weights[i].rows, network->weights[i].cols);
        init_matrix(&updates.nabla_b[i], network->biases[i].rows, network->biases[i].cols);
    }

    zero_matrix(updates.nabla_w);
    zero_matrix(updates.nabla_b);

    Matrix expected;
    init_matrix(&expected, network->sizes[network->num_layers - 1], 1);
    for (int i = 0; i < network->sizes[network->num_layers - 1]; i++) {
        expected.data[i][0] = 0;
        if (i == label) {
            expected.data[i][0] = 1;
        }
    }

    // Feedforward.

    Matrix activations[network->num_layers];
    activations[0] = *input;
    Matrix zs[network->num_layers - 1];

    for (int i = 0; i < network->num_layers - 1; i++) {
        init_matrix(&zs[i], network->weights[i].rows, input->cols);
        init_matrix(&activations[i + 1], network->weights[i].rows, input->cols);

        mul_matrix(&network->weights[i], input, &zs[i]);
        add_matrix(&zs[i], &network->biases[i], &activations[i + 1]);
        sigmoid_matrix(&activations[i + 1], &activations[i + 1]);

        free_matrix(input);
        *input = activations[i + 1];
    }

    // Backward pass.
    
    Matrix delta;
    init_matrix(&delta, network->sizes[network->num_layers - 1], 1);
    Matrix cost_derivative;
    init_matrix(&cost_derivative, network->sizes[network->num_layers - 1], 1);

    sub_matrix(&activations[network->num_layers - 1], &expected, &cost_derivative); 
    hadamard_matrix(&cost_derivative, &activations[network->num_layers - 1], &delta);

    free_matrix(&cost_derivative);

    updates.nabla_b[network->num_layers - 2] = delta;

    Matrix transposed;
    transpose_matrix(&activations[network->num_layers - 2], &transposed);
    mul_matrix(&delta, &transposed, &updates.nabla_w[network->num_layers - 2]);
    free_matrix(&transposed);

    for (int i = network->num_layers - 2; i > 0; i--) {
        Matrix z = zs[i - 1];

        Matrix sp;
        init_matrix(&sp, z.rows, z.cols);
        sigmoid_derivative_matrix(&z, &sp);

        Matrix transposed;
        transpose_matrix(&network->weights[i], &transposed);

        mul_matrix(&transposed, &delta, &delta);
        hadamard_matrix(&delta, &sp, &delta);

        free_matrix(&sp);
        free_matrix(&transposed);

        updates.nabla_b[i - 1] = delta;
        mul_matrix(&delta, &activations[i - 1], &updates.nabla_w[i - 1]);
    }

    return updates;
}

void process_batch(struct Network *network, float data_train[num_pixels][num_train], int labels_train[num_train], int batch_start, int batch_size, float eta) {
}


void accuracy(struct Network *network, float data_test[num_pixels][num_test], int labels_test[num_test]) {
}

void sgd(struct Network *network, float data_train[num_pixels][num_train], int labels_train[num_train], float data_test[num_pixels][num_test], int labels_test[num_test], int epochs, int mini_batch_size, float eta) {
    for (int i = 0; i < epochs; i++) {
        shuffle(num_pixels, num_train, data_train);
        int batch_progress = 0;
        while (batch_progress < num_train) {
            printf("Epoch %d\n", i);
            process_batch(network, data_train, labels_train, batch_progress, mini_batch_size, eta);
            batch_progress += num_train;
            accuracy(network, data_test, labels_test);
        }
    }
}
