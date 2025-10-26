#include "ap_int.h"
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>

typedef ap_uint<192>            MemIntType;
typedef hls::stream<MemIntType> MemStream;

#define BLOCK_SIZE          24
#define TILE_SIZE           6     

#define INPUT_PACK_SIZE     6
#define OUTPUT_PACK_SIZE    6

#define PARALLEL_M          6
#define PARALLEL_N          6

void StrassenReadAB(
    ap_uint<48>* input_A,
    ap_uint<48>* input_B,
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B
){
    ap_uint<48> buffer_a[BLOCK_SIZE*BLOCK_SIZE/INPUT_PACK_SIZE];
    ap_uint<48> buffer_b[BLOCK_SIZE*BLOCK_SIZE/INPUT_PACK_SIZE];

    #pragma HLS DATAFLOW
    read_AB:
    for(int i = 0; i < BLOCK_SIZE*BLOCK_SIZE/INPUT_PACK_SIZE; i++){
        #pragma HLS UNROLL factor=6
        buffer_a[i] = input_A[i];
        buffer_b[i] = input_B[i];
    }
}

void unpack_ap_uint48(ap_uint<48> packed_data, int8_t unpacked[6]) {
    #pragma HLS INLINE
    for(int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        unpacked[i] = (int8_t)(packed_data.range(8*i+7, 8*i));
    }
}

ap_uint<48> pack_int8_to_ap_uint48(int32_t data[6]) {
    #pragma HLS INLINE
    ap_uint<48> packed_data = 0;
    for(int i = 0; i < 6; i++) {
        #pragma HLS UNROLL
        packed_data.range(8*i+7, 8*i) = (ap_uint<8>)data[i];
    }
    return packed_data;
}

void base_mm(
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B,
    hls::stream<ap_uint<48>>& stream_M 
){
    int8_t A_current[TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=A_current complete
    int8_t B_current[TILE_SIZE];  
    #pragma HLS ARRAY_PARTITION variable=B_current complete
    
    int8_t M_accum[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=M_accum complete dim=0

    init:
    for(int i = 0; i < TILE_SIZE; i++) {
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            M_accum[i][j] = 0;
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
                M_accum[i][j] += (int8_t)A_current[i] * B_current[j];
            }
        }
    }

    shift:
    for(int i = 0; i < TILE_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        int32_t temp_data[TILE_SIZE];
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            temp_data[j] = M_accum[i][j];
        }
        ap_uint<48> packed_C = pack_int8_to_ap_uint48(temp_data);
        stream_M.write(packed_C);
    }
    
}

void bufferTileStrassen_1(
    hls::stream<ap_uint<48>>& stream_M,
    unsigned int idx1,
    const bool sign1,
    int32_t* buffer_c
){
    #pragma HLS INLINE OFF

    for(int p = 0; p < TILE_SIZE*TILE_SIZE; p+=INPUT_PACK_SIZE){
        #pragma HLS PIPELINE
        ap_uint<48> val = stream_M.read();
        for(int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            if(sign1){
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
        }
    }
}

void bufferTileStrassen_2(
    hls::stream<ap_uint<48>>& stream_M,
    unsigned int idx1,
    const bool sign1,
    unsigned int idx2,
    const bool sign2,
    int32_t* buffer_c
){
    #pragma HLS INLINE OFF

    for(int p = 0; p < TILE_SIZE*TILE_SIZE; p+=INPUT_PACK_SIZE){
        #pragma HLS PIPELINE
        ap_uint<48> val = stream_M.read();
        for(int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            if(sign1){
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
            if(sign2){
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
        }
    }
}

void bufferTileStrassen_3(
    hls::stream<ap_uint<48>>& stream_M,
    unsigned int idx1,
    const bool sign1,
    unsigned int idx2,
    const bool sign2,
    unsigned int idx3,
    const bool sign3,
    int32_t* buffer_c
){
    #pragma HLS INLINE OFF

    for(int p = 0; p < TILE_SIZE*TILE_SIZE; p+=INPUT_PACK_SIZE){
        #pragma HLS PIPELINE
        ap_uint<48> val = stream_M.read();
        for(int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            if(sign1){
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
            if(sign2){
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
            if(sign3){
                buffer_c[idx3*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx3*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
        }
    }
}

void bufferTileStrassen_4(
    hls::stream<ap_uint<48>>& stream_M,
    unsigned int idx1,
    const bool sign1,
    unsigned int idx2,
    const bool sign2,
    unsigned int idx3,
    const bool sign3,
    unsigned int idx4,
    const bool sign4,
    int32_t* buffer_c
){
    #pragma HLS INLINE OFF

    for(int p = 0; p < TILE_SIZE*TILE_SIZE; p+=INPUT_PACK_SIZE){
        #pragma HLS PIPELINE
        ap_uint<48> val = stream_M.read();
        for(int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            if(sign1){
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
            if(sign2){
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
            if(sign3){
                buffer_c[idx3*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx3*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
            if(sign4){
                buffer_c[idx4*TILE_SIZE*TILE_SIZE+p+i] += val.range(8*i+7, 8*i);
            }else{
                buffer_c[idx4*TILE_SIZE*TILE_SIZE+p+i] -= val.range(8*i+7, 8*i);
            }
        }
    }
}

void StrassenBlock(
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B,
    hls::stream<ap_uint<48>>& stream_M 
){
    // 49次乘法
    inner_block:
    for(int i = 0; i < 49; i++){
        base_mm(stream_A, stream_B, stream_M);
    }
}

// stream_m: 49*Tile*Tile
// stream_c: 16*Tile*Tile
void StrassenOutBuffer(
    hls::stream<ap_uint<48>>& stream_M,
    hls::stream<ap_uint<192>>& stream_C
){
    #pragma HLS DATAFLOW

    int32_t buffer_c[16*TILE_SIZE*TILE_SIZE];
    #pragma HLS BIND_STORAGE variable = buffer_c type = RAM_2P impl = BRAM
    #pragma HLS ARRAY_PARTITION variable=buffer_c complete

    bufferTileStrassen_4(stream_M, 0, 1, 3, 1, 12, 1, 15, 1, buffer_c);
    bufferTileStrassen_4(stream_M, 2, 1, 3, 0, 14, 1, 15, 0, buffer_c);
    bufferTileStrassen_2(stream_M, 1, 1, 13, 1, buffer_c);
    bufferTileStrassen_4(stream_M, 0, 1, 2, 1, 12, 1, 14, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 1, 1, 13, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 3, 1, 15, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 0, 1, 12, 1, buffer_c);
    bufferTileStrassen_4(stream_M, 8, 1, 11, 1, 12, 0, 15, 0, buffer_c);
    bufferTileStrassen_4(stream_M, 10, 1, 11, 0, 14, 0, 15, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 9, 1, 13, 0, buffer_c);
    bufferTileStrassen_4(stream_M, 8, 1, 10, 1, 12, 0, 14, 0, buffer_c);
    bufferTileStrassen_2(stream_M, 9, 1, 13, 0, buffer_c);
    bufferTileStrassen_2(stream_M, 11, 1, 15, 0, buffer_c);
    bufferTileStrassen_2(stream_M, 8, 1, 12, 0, buffer_c);
    bufferTileStrassen_2(stream_M, 4, 1, 7, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 6, 1, 7, 0, buffer_c);
    bufferTileStrassen_1(stream_M, 5, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 4, 1, 6, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 5, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 7, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 4, 1, buffer_c);
    bufferTileStrassen_4(stream_M, 0, 1, 3, 1, 8, 1, 11, 1, buffer_c);
    bufferTileStrassen_4(stream_M, 2, 1, 3, 0, 10, 1, 11, 0, buffer_c);
    bufferTileStrassen_2(stream_M, 1, 1, 9, 1, buffer_c);
    bufferTileStrassen_4(stream_M, 0, 1, 2, 1, 8, 1, 10, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 1, 1, 9, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 3, 1, 11, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 0, 1, 8, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 4, 1, 7, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 6, 1, 7, 0, buffer_c);
    bufferTileStrassen_1(stream_M, 5, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 4, 1, 6, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 5, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 7, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 4, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 12, 1, 15, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 14, 1, 15, 0, buffer_c);
    bufferTileStrassen_1(stream_M, 13, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 12, 1, 14, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 13, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 15, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 12, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 0, 1, 3, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 2, 1, 3, 0, buffer_c);
    bufferTileStrassen_1(stream_M, 1, 1, buffer_c);
    bufferTileStrassen_2(stream_M, 0, 1, 2, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 1, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 3, 1, buffer_c);
    bufferTileStrassen_1(stream_M, 0, 1, buffer_c);

    // Stream out the buffer
    for(int p = 0; p < 16*TILE_SIZE*TILE_SIZE; p+=OUTPUT_PACK_SIZE){
        #pragma HLS PIPELINE
        ap_uint<192> current_data;
        for(int i = 0; i < OUTPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            current_data.range(32*i+31, 32*i) = (ap_uint<32>)buffer_c[p+i];
        }
        stream_C << current_data;
    }
}

void StrassenWriteC(
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
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC depth=BLOCK_SIZE*BLOCK_SIZE/8

    #pragma HLS DATAFLOW

    hls::stream<ap_uint<48>> buf_A;
    hls::stream<ap_uint<48>> buf_B;
    hls::stream<ap_uint<48>> buf_M;
    hls::stream<ap_uint<192>> buf_C;

    #pragma HLS STREAM variable=buf_A depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS STREAM variable=buf_B depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS STREAM variable=buf_M depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS STREAM variable=buf_C depth=BLOCK_SIZE*BLOCK_SIZE

    StrassenReadAB(A, B, buf_A, buf_B);
    StrassenBlock(buf_A, buf_B, buf_M);
    StrassenOutBuffer(buf_M, buf_C);
    StrassenWriteC(buf_C, C);

}