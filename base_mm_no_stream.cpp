#include "ap_int.h"
#include <stdio.h>
#include <iostream>
#include "base_mm_no_stream.hpp"

// 宏定义常量


void matrix_mul(
    int8_t A[BLOCK_SIZE][BLOCK_SIZE],
    int8_t B[BLOCK_SIZE][BLOCK_SIZE],
    int32_t AB[BLOCK_SIZE][BLOCK_SIZE]
){
    #pragma HLS ARRAY_RESHAPE variable=A complete dim=2
    #pragma HLS ARRAY_RESHAPE variable=B complete dim=1
    row_for_AB:
    for(int i = 0; i < BLOCK_SIZE; ++i){
        col_for_AB:
        for(int j = 0; j < BLOCK_SIZE; ++j){
            #pragma HLS PIPELINE II=1
            int32_t sum = 0;
            product:
            for(int k = 0; k < BLOCK_SIZE; ++k){
                sum += A[i][k] * B[k][j];
            }
            AB[i][j] = sum;
        }
    }
}
