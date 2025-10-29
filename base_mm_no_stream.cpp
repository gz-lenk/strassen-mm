#include "ap_int.h"
#include <stdio.h>
#include <iostream>
#include "base_mm_no_stream.hpp"

// 宏定义常量


void matrix_mul(
    int8_t A[M][K],
    int8_t B[K][N],
    int32_t AB[M][N]
){
    #pragma HLS ARRAY_RESHAPE variable=A complete dim=2
    #pragma HLS ARRAY_RESHAPE variable=B complete dim=1
    row_for_AB:
    for(int i = 0; i < M; ++i){
        col_for_AB:
        for(int j = 0; j < N; ++j){
            #pragma HLS PIPELINE II=1
            int32_t sum = 0;
            product:
            for(int k = 0; k < K; ++k){
                sum += A[i][k] * B[k][j];
            }
            AB[i][j] = sum;
        }
    }
}
