#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

// Samples from a normal distribution using Marsaglia polar method.
float randn() {
    static bool hasSpare = false;
    static float spare;
    if(hasSpare) {
        hasSpare = false;
        return spare;
    } else {
        hasSpare = true;
        float u, v, s;
        do {
            u = (rand() / ((float) RAND_MAX)) * 2.0 - 1.0;
            v = (rand() / ((float) RAND_MAX)) * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1 || s == 0);
        s = sqrt(-2.0 * log(s) / s);
        spare = v * s;
        return u * s;
    }
}

void shuffle(int *array, size_t n) {
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
