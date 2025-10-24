#include "ap_int.h"
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include "hls_vector.h"

typedef int8_t          DTYPE_IN;
typedef int32_t         DTYPE_OUT;

typedef ap_uint<64>    AXI_DATA_IN;
typedef ap_uint<64>    AXI_DATA_OUT;

#define MATRIX_SIZE         512
#define BLOCK_SIZE          32
#define TILE_SIZE           8

#define INPUT_PACK_SIZE     8              // 256bit/8bit 输入接口一次传输的数据量
#define OUTPUT_PACK_SIZE    2               // 256bit/32bit 输出接口一次传输的数据量

#define TOTAL_MATRIX_ELEMENTS       MATRIX_SIZE*MATRIX_SIZE
#define TOTAL_BLOCK_ELEMENTS        BLOCK_SIZE*BLOCK_SIZE
#define TOTAL_TILE_ELEMENTS         TILE_SIZE*TILE_SIZE


void mm_pipeline(
    AXI_DATA_IN* A,
    AXI_DATA_IN* B,
    AXI_DATA_IN* C
);

