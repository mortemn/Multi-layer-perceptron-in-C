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

void free_network (Network network) {
    for (int i = 0; i < network.num_layers - 1; i++) {
        free_matrix(&network.biases[i]);
        free_matrix(&network.weights[i]);
    }

    free(network.biases);
    free(network.weights);
    free(network.sizes);
}

void free_delta(Delta delta, int n) {
    for (int i = 0; i < n; i++) {
        free_matrix(&delta.nabla_w[i]);
        free_matrix(&delta.nabla_b[i]);
    }

    free(delta.nabla_w);
    free(delta.nabla_b);
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

int get_prediction(Matrix *m) {
    int max = 0;
    for (int i = 1; i < m->rows; i++) {
        if (m->data[i][0] > m->data[max][0]) {
            max = i;
        }
    }
    return max;
}

Delta backprop(struct Network *network, Matrix *input, int label) {
    Delta updates;
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

    Matrix activation;
    activation = activations[0];

    for (int i = 0; i < network->num_layers - 1; i++) {
        init_matrix(&zs[i], network->weights[i].rows, activations[i].cols);

        mul_matrix(&network->weights[i], &activations[i], &zs[i]);
        add_matrix(&zs[i], &network->biases[i], &zs[i]);

        init_matrix(&activations[i + 1], zs[i].rows, zs[i].cols);
        sigmoid_matrix(&zs[i], &activations[i + 1]);
    }

    // Backward pass.
    
    Matrix delta;
    init_matrix(&delta, activations[network->num_layers - 1].rows, activations[network->num_layers - 1].cols);
    Matrix cost_derivative;
    init_matrix(&cost_derivative, activations[network->num_layers - 1].rows, activations[network->num_layers - 1].cols);

    Matrix sp;
    init_matrix(&sp, activations[network->num_layers - 1].rows, activations[network->num_layers - 1].cols);
    sigmoid_derivative_matrix(&zs[network->num_layers - 2], &sp);

    sub_matrix(&activations[network->num_layers - 1], &expected, &cost_derivative); 
    hadamard_matrix(&cost_derivative, &sp, &delta);

    free_matrix(&expected);
    free_matrix(&cost_derivative);
    free_matrix(&sp);

    updates.nabla_b[network->num_layers - 1] = delta;

    Matrix transposed;
    transpose_matrix(&activations[network->num_layers - 2], &transposed);
    mul_matrix(&delta, &transposed, &updates.nabla_w[network->num_layers - 1]);

    free_matrix(&transposed);

    for (int i = network->num_layers - 2; i > 0; i--) {
        Matrix z = zs[i - 1];

        Matrix sp;
        init_matrix(&sp, z.rows, z.cols);

        // TODO: Find if sigmoid prime is supposed to be this small (returning 0's).
        sigmoid_derivative_matrix(&z, &sp);

        Matrix transposed;
        transpose_matrix(&network->weights[i], &transposed);

        mul_mod_matrix(&transposed, &delta);
        hadamard_matrix(&delta, &sp, &delta);

        free_matrix(&sp);
        free_matrix(&transposed);
        free_matrix(&z);

        updates.nabla_b[i - 1] = delta;
        transpose_matrix(&activations[i - 1], &transposed);
        mul_matrix(&delta, &transposed, &updates.nabla_w[i - 1]);

        free_matrix(&transposed);
    }

    return updates;
}

void process_batch(struct Network *network, float data_train[num_pixels][num_train], int labels_train[num_train], int batch_start, int batch_size, float eta) {
    printf("%s\n", "Processing batch...");
    Delta delta;
    delta.nabla_w = malloc((network->num_layers - 1) * sizeof(struct Matrix));
    delta.nabla_b = malloc((network->num_layers - 1) * sizeof(struct Matrix));

    for (int i = 0; i < network->num_layers - 1; i++) {
        init_matrix(&delta.nabla_w[i], network->weights[i].rows, network->weights[i].cols);
        zero_matrix(&delta.nabla_w[i]);

        init_matrix(&delta.nabla_b[i], network->biases[i].rows, network->biases[i].cols);
        zero_matrix(&delta.nabla_b[i]);
    }

    for (int i = 0; i < batch_size; i++) {
        Matrix sample;
        index_input(data_train, batch_start + i, &sample);

        Delta n_delta = backprop(network, &sample, batch_start + i);

        for (int j = 0; j < network->num_layers - 1; j++) {
            add_matrix(&n_delta.nabla_b[j], &delta.nabla_b[j], &delta.nabla_b[j]);
            add_matrix(&n_delta.nabla_w[j], &delta.nabla_w[j], &delta.nabla_w[j]);
        }

        // TODO: Figure out why freeing n_delta affects delta.
        // free_delta(n_delta, network->num_layers - 1);
        free_matrix(&sample);
    }
    
    // Gradient descent algorithm on network weights and biases.

    for (int i = 0; i < network->num_layers - 1; i++) {
        scalar_mul_matrix(&delta.nabla_w[i], eta/batch_size, &delta.nabla_w[i]);
        sub_matrix(&network->weights[i], &delta.nabla_w[i], &network->weights[i]);

        scalar_mul_matrix(&delta.nabla_b[i], eta/batch_size, &delta.nabla_b[i]);
        sub_matrix(&network->biases[i], &delta.nabla_b[i], &network->biases[i]);
    }
    free_delta(delta, network->num_layers - 1);
}


void accuracy(struct Network *network, float data_test[num_pixels][num_test], int labels_test[num_test]) {
    printf("%s\n", "Calculating accuracy...");
    int correct = 0;
    for (int i = 0; i < num_test; i++) {
        Matrix input;
        Matrix output;

        index_input(data_test, i, &input);
        forward_prop(network, &input, &output);

        int n = get_prediction(&output);
        if (labels_test[i] == n) correct += 1; 

        free_matrix(&output);
    }
    float accuracy = (float)correct/(float)num_test;
    printf("Accuracy: %f\n", accuracy);
}

void sgd(struct Network *network, float data_train[num_pixels][num_train], int labels_train[num_train], float data_test[num_pixels][num_test], int labels_test[num_test], int epochs, int mini_batch_size, float eta) {
    for (int i = 0; i < epochs; i++) {
        shuffle(num_pixels, num_train, data_train);
        int batch_progress = 0;
        while (batch_progress < num_train) {
            printf("Epoch %d\n", i+1);
            process_batch(network, data_train, labels_train, batch_progress, mini_batch_size, eta);
            batch_progress += num_train;
            accuracy(network, data_test, labels_test);
        }
    }
}
