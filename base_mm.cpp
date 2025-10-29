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
    for(int i = 0; i < BLOCK_SIZE; i++){
        for(int j = 0; j < BLOCK_SIZE/6; j++){
            #pragma HLS PIPELINE
            stream_A.write(input_A[i*(BLOCK_SIZE/6)+j]);
        }
    }
    read_B:
    for(int j = 0; j < BLOCK_SIZE; j++){
        for(int i = 0; i < BLOCK_SIZE/6; i++){
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
    #pragma HLS ALLOCATION instances=mul limit=36 function

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
        ap_uint<48> packed_A = stream_A.read();
        ap_uint<48> packed_B = stream_B.read();
        
        unpack_ap_uint48(packed_A, A_current);
        unpack_ap_uint48(packed_B, B_current);

        for(int i = 0; i < TILE_SIZE; i++) {
            #pragma HLS UNROLL
            for(int j = 0; j < TILE_SIZE; j++) {
                #pragma HLS UNROLL
                int32_t mult_result = (int32_t)A_current[i] * B_current[j];
                C_accum[i][j] += mult_result;
            }
        }
    }

    shift:
    for(int i = 0; i < TILE_SIZE; i++) {
        int32_t temp_data[TILE_SIZE];
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            temp_data[j] = C_accum[i][j];
        }
        ap_uint<192> packed_C = pack_int32_to_ap_uint192(temp_data);
        stream_C.write(packed_C);
    }
    
}

void base_mm(
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B,
    hls::stream<ap_uint<192>>& stream_C 
){
    int32_t C_accum[TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=C_accum complete dim=0

    row_for_AB:
    for(int i = 0; i < TILE_SIZE; ++i){
        col_for_AB:
        for(int j = 0; j < TILE_SIZE; ++j){
            ap_uint<48> packed_A = stream_A.read();
            ap_uint<48> packed_B = stream_B.read();

            int8_t A[TILE_SIZE], B[TILE_SIZE];
            unpack_ap_uint48(packed_A, A);
            unpack_ap_uint48(packed_B, B);

            int32_t sum = 0;
            product:
            for(int k = 0; k < TILE_SIZE; ++k){
                sum += A[k] * B[k];
            }
            C_accum[j] = sum;
        }
        ap_uint<192> packed_C = pack_int32_to_ap_uint192(C_accum);
        stream_C.write(packed_C);
    }
}

void base_mm_systolic(
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B,
    hls::stream<ap_uint<192>>& stream_C 
){
    int8_t A_systolic[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=A_systolic complete dim=1
    #pragma HLS ARRAY_PARTITION variable=A_systolic complete dim=2
    
    int8_t B_systolic[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=B_systolic complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_systolic complete dim=2
    
    int32_t C_accum[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=C_accum complete dim=0

    systolic_init:
    for(int i = 0; i < TILE_SIZE; i++) {
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            C_accum[i][j] = 0;
            A_systolic[i][j] = 0;
            B_systolic[i][j] = 0;
        }
    }

    systolic_compute:
    for(int step = 0; step < TILE_SIZE * 2 - 1; step++) {
        #pragma HLS PIPELINE II=1

        if(step < TILE_SIZE) {
            ap_uint<48> packed_A = stream_A.read();
            ap_uint<48> packed_B = stream_B.read();
            
            int8_t A_new[TILE_SIZE], B_new[TILE_SIZE];
            unpack_ap_uint48(packed_A, A_new);
            unpack_ap_uint48(packed_B, B_new);
            
            for(int i = 0; i < TILE_SIZE; i++) {
                #pragma HLS UNROLL
                A_systolic[i][0] = A_new[i];
                B_systolic[0][i] = B_new[i];
            }
        }
        
        for(int i = 0; i < TILE_SIZE; i++) {
            #pragma HLS UNROLL
            for(int j = TILE_SIZE-1; j > 0; j--) {
                #pragma HLS UNROLL
                A_systolic[i][j] = A_systolic[i][j-1];
                B_systolic[j][i] = B_systolic[j-1][i];
            }
        }
        
        for(int i = 0; i < TILE_SIZE; i++) {
            #pragma HLS UNROLL
            for(int j = 0; j < TILE_SIZE; j++) {
                #pragma HLS UNROLL
                if(i + j <= step && step < i + j + TILE_SIZE) {
                    C_accum[i][j] += (int32_t)A_systolic[i][j] * B_systolic[i][j];
                }
            }
        }
    }
    
    for(int i = 0; i < TILE_SIZE; i++) {
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
    #pragma HLS DATAFLOW

    hls::stream<ap_uint<192>> tile_C_stream;
    #pragma HLS STREAM variable=tile_C_stream depth=BLOCK_SIZE*BLOCK_SIZE/TILE_SIZE

    

    // 分块矩阵乘法：C[i][j] += A[i][k] * B[k][j]
    // 写入临时流
    block_i_loop:
    for(int block_i = 0; block_i < BLOCK_SIZE; block_i += TILE_SIZE) {
        block_j_loop:
        for(int block_j = 0; block_j < BLOCK_SIZE; block_j += TILE_SIZE) {
            block_k_loop:
            for(int block_k = 0; block_k < BLOCK_SIZE; block_k += TILE_SIZE) {
                #pragma HLS PIPELINE 
                base_mm_systolic(stream_A, stream_B, tile_C_stream);
                //base_mm(stream_A, stream_B, tile_C_stream);
            }
        }
    }

    // 累加临时流
    // Stream: A[i][0] * B[0][j], A[i][1] * B[1][j], A[i][2] * B[2][j] ...
    accum_i:
    for(int block_i = 0; block_i < BLOCK_SIZE; block_i += TILE_SIZE) {
        #pragma HLS LOOP_TRIPCOUNT max=4 min=4 avg=4
        accum_j:
        for(int block_j = 0; block_j < BLOCK_SIZE; block_j += TILE_SIZE) {
            #pragma HLS PIPELINE
            #pragma HLS LOOP_TRIPCOUNT max=4 min=4 avg=4
            int32_t accum_C[TILE_SIZE][TILE_SIZE];
            #pragma HLS ARRAY_PARTITION variable=accum_C complete dim=0

            accum_k:
            for(int block_k = 0; block_k < BLOCK_SIZE; block_k += TILE_SIZE) {
                #pragma HLS LOOP_TRIPCOUNT max=4 min=4 avg=4
                for(int tile_row = 0; tile_row < TILE_SIZE; tile_row++){
                    ap_uint<192> pack_tile_c = tile_C_stream.read();
                    int32_t pack_data[TILE_SIZE];
                    unpack_ap_uint192(pack_tile_c, pack_data);
                    
                    for(int i = 0; i < TILE_SIZE; i++){
                        #pragma HLS UNROLL
                        accum_C[tile_row][i] += pack_data[i];
                    }
                }
            }
            
            write_to_stream:
            for(int tile_row = 0; tile_row < TILE_SIZE; tile_row++) {
                #pragma HLS LOOP_TRIPCOUNT max=6 min=6 avg=6
                int32_t temp_data[TILE_SIZE];
                for(int j = 0; j < TILE_SIZE; j++) {
                    #pragma HLS UNROLL
                    temp_data[j] = accum_C[tile_row][j];
                }
                ap_uint<192> packed_C = pack_int32_to_ap_uint192(temp_data);
                stream_C.write(packed_C);
            }
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

    #pragma HLS DATAFLOW

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
