#include <stdlib.h>
#include "math_utils.h"

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
