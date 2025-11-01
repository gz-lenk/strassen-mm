#include "ap_int.h"
#include "hls_stream.h"
#include "hls_vector.h"
#include <stdio.h>
#include <iostream>

#define INPUT_PACK_SIZE     24
#define OUTPUT_PACK_SIZE    6

#define BLOCK_SIZE          24
#define TILE_SIZE           6     

typedef int8_t          DTYPE_IN;
typedef int32_t         DTYPE_OUT;

typedef ap_uint<192>                             MemIntType;
typedef hls::vector<DTYPE_IN, INPUT_PACK_SIZE>   MemVecType;
typedef hls::stream<MemVecType>                  MemStream;

void GemmReadAB(
    MemIntType* input_A,
    MemIntType* input_B,
    MemStream&  stream_A,
    MemStream& stream_B
){
    #pragma HLS DATAFLOW

    read_A_loop:
    for(int block_row = 0; block_row < BLOCK_SIZE; block_row+=TILE_SIZE){
        for(int tile_row = 0; tile_row < TILE_SIZE; tile_row++){
            #pragma HLS PIPELINE
            int element_idx = block_row + tile_row;
            MemIntType row_a = input_A[element_idx];
            MemVecType row_vec_a;
            for(int i = 0; i < INPUT_PACK_SIZE; i++){
                #pragma HLS UNROLL
                row_vec_a[i] = row_a.range(8*i+7, 8*i);
            }
            stream_A.write(row_vec_a);
        }
    }

    read_B_loop:
    for(int block_col = 0; block_col < BLOCK_SIZE; block_col+=TILE_SIZE){
        for(int tile_col = 0; tile_col < TILE_SIZE; tile_col++){
            #pragma HLS PIPELINE
            int element_idx = block_col + tile_col;
            MemIntType col_b = input_B[element_idx];
            MemVecType col_vec_b;
            for(int i = 0; i < INPUT_PACK_SIZE; i++){
                #pragma HLS UNROLL
                col_vec_b[i] = col_b.range(8*i+7, 8*i);
            }
            stream_B.write(col_vec_b);
        }
    }
}



void mac(
    MemVecType buffer_A[BLOCK_SIZE],
    MemStream& stream_B,
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C
){
    MemVecType buffer_B[TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=buffer_B complete dim=1
    hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> vec_C;

    read_B:
    for(int block_col = 0; block_col < TILE_SIZE; block_col++){
        #pragma HLS PIPELINE
        buffer_B[block_col] = stream_B.read();
    }

    tile_mac:
    for(int block_row = 0; block_row < BLOCK_SIZE; block_row ++){
        #pragma HLS PIPELINE
        for(int tile_col = 0; tile_col < TILE_SIZE; tile_col++){
            #pragma HLS UNROLL
            DTYPE_OUT sum = 0;
            for(int k = 0; k < INPUT_PACK_SIZE; k++){
                sum += (int32_t)buffer_A[block_row][k] * buffer_B[tile_col][k];
            }
            vec_C[tile_col] = sum;
        }
        write_C:
        stream_C << vec_C;
    }

}

void GemmBlock(
    MemStream& stream_A,
    MemStream& stream_B,
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C
){
    #pragma HLS DATAFLOW

    MemVecType buffer_A[BLOCK_SIZE];
    #pragma HLS ARRAY_PARTITION variable=buffer_A complete dim=1

    read_A:
    for(int block_row = 0; block_row < BLOCK_SIZE; block_row++){
        buffer_A[block_row] = stream_A.read();
    }

    mac:
    for(int block_col = 0; block_col < BLOCK_SIZE; block_col+=TILE_SIZE){
        #pragma HLS PIPELINE
        mac(buffer_A, stream_B, stream_C);
    }

}

// 将打包的hls::vector输出流写入内存
void GemmWriteC(
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C,
    MemIntType* output_C
){
    #pragma HLS DATAFLOW

    const int total_pack = (BLOCK_SIZE * BLOCK_SIZE) / OUTPUT_PACK_SIZE;
    
    write_C:
    for(int block_i = 0; block_i < total_pack; block_i++) {
        #pragma HLS LOOP_TRIPCOUNT max=96 min=96 avg=96
        #pragma HLS PIPELINE II=1
        hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> packed_data = stream_C.read();
        MemIntType output_data = 0;
        for(int i = 0; i < OUTPUT_PACK_SIZE; i++) {
            #pragma HLS UNROLL
            output_data.range(32*i+31, 32*i) = (ap_uint<32>)packed_data[i];
        }
        output_C[block_i] = output_data;
    }
}


void mm_pipeline(
    MemIntType* A,
    MemIntType* B,
    MemIntType* C
){
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC depth=BLOCK_SIZE*BLOCK_SIZE

    #pragma HLS DATAFLOW

    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>> buf_A;
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>> buf_B;
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>> buf_C;

    #pragma HLS STREAM variable=buf_A depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS STREAM variable=buf_B depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS STREAM variable=buf_C depth=BLOCK_SIZE*BLOCK_SIZE

    GemmReadAB(A, B, buf_A, buf_B);
    GemmBlock(buf_A, buf_B, buf_C);
    GemmWriteC(buf_C, C);

}
