#include "csv_reader.h"
#include "network.h"
#include "math_utils.h"
#include <stdio.h>
#include <time.h>

#define MAXCHAR 10000

int num_pixels = 784;
int num_train = 60000;
int num_test = 10000;

float data_train[784][60000];
float data_test[784][10000];
int labels_train[60000];
int labels_test[10000];

int sizes[3] = {784, 30, 10};

int main() {
    srand(time(NULL));
    load_data_train(num_train, num_pixels, data_train, labels_train);
    load_data_test(num_test, num_pixels, data_test, labels_test);

    Network network;
    init_network(&network, 3, sizes);

    Matrix input;
    init_matrix(&input, num_pixels, 1);

    Matrix output;

    for (int i = 0; i < num_pixels; i++) {
        input.data[i][0] = data_test[i][0];
    }

    sgd(&network, data_train, labels_train, data_test, labels_test, 30, 10, 3.0);

    return 0;
}
