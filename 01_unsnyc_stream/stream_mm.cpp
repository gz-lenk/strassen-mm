#include "ap_int.h"
#include <stdio.h>
#include <iostream>
#include <hls_stream.h>
#include "stream_mm.hpp"

void read_A(
    const AXI_DATA_IN  *A,
    hls::stream<int8_block_t>& A_stream
){
    for (int br = 0; br < FACTOR; br++) { 
        for (int k = 0; k < FACTOR; k++){
            for (int bc = 0; bc < FACTOR; bc++) {
                #pragma HLS PIPELINE
                int8_block_t temp_block;

                for (int pack_idx = 0; pack_idx < BLOCK_ELEMENT/PACK_SIZE; pack_idx++){
                    int global_row = br * BLOCK_SIZE + pack_idx;
                    int global_col = bc;
                    ap_int<48> temp_data = A[global_row * FACTOR + global_col];
                    for(int i = 0; i < PACK_SIZE; i++){
                        temp_block.data[pack_idx][i] = temp_data.range(8*i+7, 8*i);
                    }
                }
                
                A_stream << temp_block;
            }
        }
    }
}

// 行优先
void read_B(
    const AXI_DATA_IN  *B,
    hls::stream<int8_block_t>& B_stream
){
    for(int k = 0; k < FACTOR; k++){
        for (int bc = 0; bc < FACTOR; bc++) { 
            for (int br = 0; br < FACTOR; br++) {
                #pragma HLS PIPELINE
                int8_block_t temp_block;

                for (int pack_idx = 0; pack_idx < BLOCK_ELEMENT/PACK_SIZE; pack_idx++){
                    int global_row = br * BLOCK_SIZE + pack_idx;
                    int global_col = bc;
                    ap_int<48> temp_data = B[global_row * FACTOR + global_col];
                    for(int i = 0; i < PACK_SIZE; i++){
                        temp_block.data[pack_idx][i] = temp_data.range(8*i+7, 8*i);
                    }
                }
                
                B_stream << temp_block;
            }
        }
    }
}

void matmul_block(
    hls::stream<int8_block_t>& A_stream,
    hls::stream<int8_block_t>& B_stream,
    hls::stream<int32_block_t>& C_stream
){
    for (int br = 0; br < FACTOR; br++) {
        for (int bc = 0; bc < FACTOR; bc++){
            int32_block_t c_block = {0};
            #pragma HLS ARRAY_PARTITION variable=c_block cyclic factor=3 dim=2
            mac:
            for (int k = 0; k < FACTOR; k++) {
                int8_block_t temp_block_A = A_stream.read();
                int8_block_t temp_block_B = B_stream.read();

                for (int i = 0; i < BLOCK_SIZE; i++) { 
                    #pragma HLS UNROLL factor=6
                    for (int j = 0; j < BLOCK_SIZE; j++) {  
                        int32_t sum = 0;
                        for (int m = 0; m < BLOCK_SIZE; m++) { 
                            sum += (int32_t)temp_block_A.data[i][m] * temp_block_B.data[m][j];
                        }
                        c_block.data[i][j] += sum;  // 改为累加而不是赋值
                    }
                }
            }
            C_stream << c_block;
        }
    }
}


void store_C(
    hls::stream<int32_block_t>& stream_C, 
    AXI_DATA_OUT* C
) {
    for (int br = 0; br < FACTOR; br++) {
        for (int bc = 0; bc < FACTOR; bc++) {
            #pragma HLS PIPELINE II = 1
            int32_block_t c_block = stream_C.read();

            for (int pack_idx = 0; pack_idx < BLOCK_ELEMENT/PACK_SIZE; pack_idx++){
                int global_row = br * BLOCK_SIZE + pack_idx;
                int global_col = bc;
                ap_int<192> temp_data = 0;
                for(int i = 0; i < PACK_SIZE; i++){
                    temp_data.range(32*i+31, 32*i) = c_block.data[pack_idx][i];
                }
                C[global_row * FACTOR + global_col] = temp_data;
            }
        }
    }
}

void matrix_mul(
    const AXI_DATA_IN  *A,
    const AXI_DATA_IN  *B,
    AXI_DATA_OUT *C
){
    #pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmemA depth = MAT_SIZE * MAT_SIZE/6 max_burst_length=16
    #pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmemB depth = MAT_SIZE * MAT_SIZE/6 max_burst_length=16
    #pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmemC depth = MAT_SIZE * MAT_SIZE/6 max_burst_length=16

    hls::stream<int8_block_t> A_stream;
    #pragma HLS STREAM variable=A_stream depth=4 type=unsync
    hls::stream<int8_block_t> B_stream;
    #pragma HLS STREAM variable=B_stream depth=4 type=unsync
    hls::stream<int32_block_t> C_stream;
    #pragma HLS STREAM variable=C_stream depth=4 type=unsync

    #pragma HLS DATAFLOW
    read_A(A, A_stream);
    read_B(B, B_stream);
    matmul_block(A_stream, B_stream, C_stream);
    store_C(C_stream, C);

}
