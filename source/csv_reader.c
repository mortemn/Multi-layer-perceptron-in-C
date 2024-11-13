#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csv_reader.h"
#include "math_utils.h"

#define MAXCHAR 10000

void load_data(int num_train, int num_pixels, int data[num_train][num_pixels], int labels[num_train], char *path) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    char row[MAXCHAR];
    char *token;

    // Skip the header line
    fgets(row, MAXCHAR, fp);

    // Load contents of CSV into the array
    int i = 0;
    while (fgets(row, MAXCHAR, fp) != NULL && i < num_train) {
        token = strtok(row, ",");
        if (token != NULL) {
            labels[i] = atoi(token); // Read label
            token = strtok(NULL, ",");
        }

        int j = 0;
        while (token != NULL && j < num_pixels) {
            data[i][j] = atoi(token); // Read pixel values
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    fclose(fp);
}

void load_data_train(int num_train, int num_pixels, float data_train[num_pixels][num_train], int labels_train[num_train]) {
    int (*data)[num_pixels] = malloc(sizeof(int[num_train][num_pixels]));
    if (data == NULL) {
        perror("Memory allocation failed");
        return;
    }

    load_data(num_train, num_pixels, data, labels_train, "mnist_train.csv");

    // Transpose data from `data` to `data_train`
    for (int i = 0; i < num_train; i++) {
        for (int j = 0; j < num_pixels; j++) {
            data_train[j][i] = (float)data[i][j] / 255.0; // Convert int to float for data_train
        }
    }

    free(data);
}

void load_data_test(int num_test, int num_pixels, float data_test[num_pixels][num_test], int labels_test[num_test]) {
    int (*data)[num_pixels] = malloc(sizeof(int[num_test][num_pixels]));
    if (data == NULL) {
        perror("Memory allocation failed");
        return;
    }

    load_data(num_test, num_pixels, data, labels_test, "mnist_test.csv");

    // Transpose data from `data` to `data_test`
    for (int i = 0; i < num_test; i++) {
        for (int j = 0; j < num_pixels; j++) {
            data_test[j][i] = (float)data[i][j] / 255.0; // Convert int to float for data_test
        }
    }

    free(data);
}

