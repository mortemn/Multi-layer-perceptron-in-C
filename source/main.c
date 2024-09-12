#include "math_utils.h"
#include "csv_reader.h"

#define MAXCHAR 10000

int num_pixels = 784;
int num_train = 60000;
float data_train[784][60000];

int sizes[3] = {784, 30, 10};

struct Weight {
    float **weights;
    int size_1;
    int size_2;
};

void init_biases(float *biases[]) {
    for (int i = 0; i < sizes[1]; i++) {
        biases[0][i] = randn();
    }

    for (int i = 0; i < sizes[2]; i++) {
        biases[1][i] = randn();
    }
}

void init_weights(struct Weight weight) {
    // for (int i = 0; i < weight.size_1; i++) {
    //     for (int j = 0; j < weight.size_2; j++) {
    //         weight.weights[i][j] = randn();
    //     }
    // }
}

int main() {
    load_data_train(num_train, num_pixels, data_train);

    float *biases[] = {malloc(sizes[1] * sizeof(float)), malloc(sizes[2] * sizeof(float))};

    float weights_10[sizes[1]][sizes[0]];
    float weights_21[sizes[2]][sizes[1]];

    float* point_w1 = &weights_10[0][0];
    float* point_w2 = &weights_21[0][0];

    struct Weight weight_1 = {&point_w1, sizes[1], sizes[0]};
    struct Weight weight_2 = {&point_w2, sizes[2], sizes[1]};

    init_biases(biases); 

    init_weights(weight_1);

    return 0;
}
