#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#define MAXCHAR 10000

int data[60000][785];
int data_dev[785][1000];
int data_train[785][59000];

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
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

    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 785; j++) {
            data_dev[j][i] = data[i][j];
        }
    }

    for (int i = 0; i < 59000; i++) {
        for (int j = 0; j < 785; j++) {
            data_train[j][i] = data[i + 1000][j];
        }
    } 

    fclose(fp);

    return 0;
}
