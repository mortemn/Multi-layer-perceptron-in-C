#include "csv_reader.h"
#include "math_utils.h"
#include "network.h"

#define MAXCHAR 10000

int num_pixels = 784;
int num_train = 60000;
float data_train[784][60000];

int sizes[3] = {784, 30, 10};

int main() {
    load_data_train(num_train, num_pixels, data_train);

    Network network;
    init_network(&network, 3, sizes);

    return 0;
}
