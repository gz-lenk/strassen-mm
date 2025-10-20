#include "ap_int.h"
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>

typedef ap_uint<192>            MemIntType;
typedef hls::stream<MemIntType> MemStream;

#define PACK_SIZE           32              // 256bit/8bit
#define PACKED_C            8               // 256bit/32bit

#define MAT_SIZE            768
#define BLOCK_SIZE          24
#define TILE_SIZE           6       

#define PARALLEL_M          6
#define PARALLEL_N          6
#define K_BUFFER_DIM        TILE_SIZE + PARALLEL_M + PARALLEL_N

void GemmReadAB_wide(
    ap_uint<48>* input_A,
    ap_uint<48>* input_B,
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B
){
    #pragma HLS DATAFLOW
    read_A:
    for(int i = 0; i < BLOCK_SIZE/6; i++){
        for(int j = 0; j < BLOCK_SIZE; j++){
            #pragma HLS PIPELINE
            stream_A.write(input_A[i*(BLOCK_SIZE/6)+j]);
        }
    }
    read_B:
    for(int j = 0; j < BLOCK_SIZE/6; j++){
        for(int i = 0; i < BLOCK_SIZE; i++){
            #pragma HLS PIPELINE
            stream_B.write(input_B[j*(BLOCK_SIZE/6)+i]);
        }
    }
}

void unpack_ap_uint48(ap_uint<48> packed_data, int8_t unpacked[6]) {
    #pragma HLS INLINE
    for(int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        unpacked[i] = (int8_t)(packed_data.range(8*i+7, 8*i));
    }
}

void unpack_ap_uint192(ap_uint<192> packed_data, int32_t unpacked[6]) {
    #pragma HLS INLINE
    for(int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        unpacked[i] = (int32_t)(packed_data.range(32*i+31, 32*i));
    }
}


ap_uint<192> pack_int32_to_ap_uint192(int32_t data[6]) {
    #pragma HLS INLINE
    ap_uint<192> packed_data = 0;
    for(int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        packed_data.range(32*i+31, 32*i) = (ap_uint<32>)data[i];
    }
    return packed_data;
}

void base_mm_shift(
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B,
    hls::stream<ap_uint<192>>& stream_C 
){
    int8_t A_current[TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=A_current complete
    int8_t B_current[TILE_SIZE];  
    #pragma HLS ARRAY_PARTITION variable=B_current complete
    
    int32_t C_accum[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=C_accum complete dim=0

    init:
    for(int i = 0; i < TILE_SIZE; i++) {
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            C_accum[i][j] = 0;
        }
    }

    mac:
    for(int k = 0; k < TILE_SIZE; k++) {
        #pragma HLS PIPELINE II=1
        
        ap_uint<48> packed_A = stream_A.read();
        ap_uint<48> packed_B = stream_B.read();
        
        unpack_ap_uint48(packed_A, A_current);
        unpack_ap_uint48(packed_B, B_current);

        for(int i = 0; i < TILE_SIZE; i++) {
            #pragma HLS UNROLL
            for(int j = 0; j < TILE_SIZE; j++) {
                #pragma HLS UNROLL
                C_accum[i][j] += (int32_t)A_current[i] * B_current[j];
            }
        }
    }

    shift:
    for(int i = 0; i < TILE_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        int32_t temp_data[TILE_SIZE];
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            temp_data[j] = C_accum[i][j];
        }
        ap_uint<192> packed_C = pack_int32_to_ap_uint192(temp_data);
        stream_C.write(packed_C);
    }


}

void GemmBlock_wide(
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B,
    hls::stream<ap_uint<192>>& stream_C 
){
    // 64次乘法
    outer_block:
    for(int block_i = 0; block_i < BLOCK_SIZE; block_i+=TILE_SIZE){
        #pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
        inner_block:
        for(int block_j = 0; block_j < BLOCK_SIZE; block_j+=TILE_SIZE){
            #pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4
            base_mm_shift(stream_A, stream_B, stream_C);
        }
    }
}

// 将打包的ap_uint<192>输出流写入内存
void GemmWriteC_wide(
    hls::stream<ap_uint<192>>& stream_C,
    ap_uint<192>* output_C
){
    #pragma HLS DATAFLOW

    const int total_blocks = (BLOCK_SIZE * BLOCK_SIZE) / 6;
    
    write_C:
    for(int block_i = 0; block_i < total_blocks; block_i++) {
        #pragma HLS PIPELINE II=1
        ap_uint<192> packed_data = stream_C.read();
        output_C[block_i] = packed_data;
    }
}


void mm_pipeline(
    ap_uint<48>* A,
    ap_uint<48>* B,
    ap_uint<192>* C
){
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC depth=BLOCK_SIZE*BLOCK_SIZE

    hls::stream<ap_uint<48>> buf_A;
    hls::stream<ap_uint<48>> buf_B;
    hls::stream<ap_uint<192>> buf_C;

    #pragma HLS STREAM variable=buf_A depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS STREAM variable=buf_B depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS STREAM variable=buf_C depth=BLOCK_SIZE*BLOCK_SIZE

    GemmReadAB_wide(A, B, buf_A, buf_B);
    GemmBlock_wide(buf_A, buf_B, buf_C);
    GemmWriteC_wide(buf_C, C);

}
