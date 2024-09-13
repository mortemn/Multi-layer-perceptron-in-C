struct Matrix {
    int rows;
    int cols;
    float **data;
};

struct Network {
    int num_layers;
    int *sizes;
    struct Matrix *biases;
    struct Matrix *weights;
};
