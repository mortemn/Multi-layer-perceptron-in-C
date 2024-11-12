#include "csv_reader.h"
#include "network.h"
#include <stdlib.h>
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

    printf("Loading CSV...\n");
    load_data_train(num_train, num_pixels, data_train, labels_train);
    load_data_test(num_test, num_pixels, data_test, labels_test);

    printf("Initializing network...\n");
    Network network;
    init_network(&network, 3, sizes);

    printf("Training network...\n");
    sgd(&network, data_train, labels_train, data_test, labels_test, 5, 10, 3.0);

    free_network(&network);
    return 0;
}
