#include <stdio.h>
#include <string.h>
#include "math_utils.h"

#define MAXCHAR 10000

int data[60000][785];
int data_train[785][60000];
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
    FILE *fp;
    char row[MAXCHAR];
    char *token;

    fp = fopen("mnist_train.csv", "r");

    // Gets first row from csv.
    fgets(row, MAXCHAR, fp);

    // Loads contents of training csv into array.
    int i = 0;
    while(!feof(fp)) {
        fgets(row, MAXCHAR, fp);
        token = strtok(row, ",");

        int j = 0;
        while(token != NULL) {
            data[i][j] = atoi(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    shuffle(data, 60000);

    for (int i = 0; i < 60000; i++) {
        for (int j = 0; j < 785; j++) {
            data_train[j][i] = data[i + 1000][j];
        }
    }

    float *biases[] = {malloc(sizes[1] * sizeof(float)), malloc(sizes[2] * sizeof(float))};

    float weights_10[sizes[1]][sizes[0]];
    float weights_21[sizes[2]][sizes[1]];

    float* point_w1 = &weights_10[0][0];
    float* point_w2 = &weights_21[0][0];

    struct Weight weight_1 = {&point_w1, sizes[1], sizes[0]};
    struct Weight weight_2 = {&point_w2, sizes[2], sizes[1]};

    init_biases(biases); 

    init_weights(weight_1);

    fclose(fp);

    return 0;
}
