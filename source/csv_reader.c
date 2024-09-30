#include <stdio.h>
#include <string.h>
#include "math_utils.h"

#define MAXCHAR 10000

void load_data(int num_train, int num_pixels, int data[num_train][num_pixels], char *path) {
    FILE *fp;
    char row[MAXCHAR];
    char *token;

    fp = fopen(path, "r");

    // Gets first row from csv.
    fgets(row, MAXCHAR, fp);

    // Loads contents of training csv into array.
    int i = 0;
    while (!feof(fp)) {
        fgets(row, MAXCHAR, fp);
        token = strtok(row, ",");

        int j = 0;
        while (token != NULL) {
            data[i][j] = atoi(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    fclose(fp);
}

void load_data_train(int num_train, int num_pixels, float data_train[num_train][num_pixels]) {
    int (*data)[num_pixels] = malloc(sizeof(int[num_train][num_pixels]));

    load_data(num_train, num_pixels, data, "mnist_train.csv");

    shuffle(*data, num_train);

    for (int i = 0; i < num_train; i++) {
        for (int j = 0; j < num_pixels; j++) {
            data_train[j][i] = data[i][j];
        }
    }

    free(data);
}

void load_data_test(int num_test, int num_pixels, float data_train[num_test][num_pixels]) {
    int (*data)[num_pixels] = malloc(sizeof(int[num_test][num_pixels]));

    load_data(num_test, num_pixels, data, "mnist_test.csv");

    shuffle(*data, num_test);

    for (int i = 0; i < num_test; i++) {
        for (int j = 0; j < num_pixels; j++) {
            data_train[j][i] = data[i][j];
        }
    }

    free(data);
}
