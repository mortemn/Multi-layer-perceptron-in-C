#ifndef NETWORK_H
#define NETWORK_H

extern int num_pixels;
extern int num_train;
extern int num_test;


typedef struct Network {
	int num_layers;
	int *sizes;
	struct Matrix *biases;
	struct Matrix *weights;
} Network;

typedef struct Updates {
	struct Matrix *nabla_w;
	struct Matrix *nabla_b;
} Updates;

void init_network(Network *network, int num_layers, int *sizes);

void forward_prop(Network *network, struct Matrix *input, struct Matrix *output);

void sgd(Network *network, float data_train[num_pixels][num_train], int labels_train[num_train], float data_test[num_pixels][num_test], int labels_test[num_test], int epochs, int mini_batch_size, float eta);

#endif
