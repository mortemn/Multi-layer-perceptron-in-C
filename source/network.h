#ifndef NETWORK_H
#define NETWORK_H

typedef struct Network {
	int num_layers;
	int *sizes;
	struct Matrix *biases;
	struct Matrix *weights;
} Network;

void init_network(Network *network, int num_layers, int *sizes);

#endif
