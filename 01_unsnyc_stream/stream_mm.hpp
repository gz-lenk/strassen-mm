#ifndef _BLOCK_MM_H 
#define _BLOCK_MM_H

#include "ap_int.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

#define DTYPE_IN        int8_t
#define DTYPE_OUT       int32_t

typedef ap_int<48>    AXI_DATA_IN;
typedef ap_int<192>    AXI_DATA_OUT;

#define MAT_SIZE 24
#define TOTAL_ELEMENT 24*24
#define BLOCK_SIZE 6
#define BLOCK_ELEMENT 6*6
#define FACTOR 4

#define PACK_SIZE 6

typedef struct {
    int8_t data[BLOCK_SIZE][BLOCK_SIZE];
} int8_block_t; 

typedef struct {
    int32_t data[BLOCK_SIZE][BLOCK_SIZE];
} int32_block_t; 

void matrix_mul(
    const AXI_DATA_IN *A,
    const AXI_DATA_IN *B,
    AXI_DATA_OUT *C
);

#endif
