#ifndef CSV_READER_H
#define CSV_READER_H

void load_data_train(int num_pixels, int num_train, float data_train[num_pixels][num_train]);

void load_data_test(int num_pixels, int num_test, float data_test[num_pixels][num_test]);

#endif
