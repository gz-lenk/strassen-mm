#include "ap_int.h"
#include "hls_stream.h"
#include "hls_vector.h"
#include <stdio.h>
#include <iostream>

#define INPUT_PACK_SIZE     6
#define OUTPUT_PACK_SIZE    6

#define PACK_SIZE           32 
#define PACKED_C            8  

#define BLOCK_SIZE          24
#define TILE_SIZE           6     

typedef int8_t          DTYPE_IN;
typedef int32_t         DTYPE_OUT;

typedef ap_uint<48>                              MemIntType;
typedef hls::vector<DTYPE_IN, INPUT_PACK_SIZE>   MemVecType;
typedef hls::stream<MemVecType>                  MemStream;

#define PARALLEL_M          6
#define PARALLEL_N          6

void GemmReadAB_wide(
    ap_uint<48>* input_A,
    ap_uint<48>* input_B,
    MemStream& stream_A,
    MemStream& stream_B
){
    #pragma HLS DATAFLOW

    read_A_loop:
    for(int block_row = 0; block_row < (BLOCK_SIZE/TILE_SIZE); block_row++){
        for(int block_col = 0; block_col < (BLOCK_SIZE/TILE_SIZE); block_col++){
            for(int loop_dim = 0; loop_dim < (BLOCK_SIZE/TILE_SIZE); loop_dim++){
                int block_idx = block_row*(BLOCK_SIZE/TILE_SIZE)+block_col;
                for(int tile_row = 0; tile_row < (TILE_SIZE*TILE_SIZE/INPUT_PACK_SIZE); tile_row++){
                    int element_idx = block_idx*(TILE_SIZE*TILE_SIZE/INPUT_PACK_SIZE)+tile_row;
                    MemIntType row_a = input_A[element_idx];
                    MemVecType row_vec_a;
                    for(int i = 0; i < INPUT_PACK_SIZE; i++){
                        #pragma HLS UNROLL
                        row_vec_a[i] = row_a.range(8*i+7, 8*i);
                    }
                    stream_A.write(row_vec_a);
                }
            }
        }
    }

    read_B_loop:
    for(int loop_dim = 0; loop_dim < (BLOCK_SIZE/TILE_SIZE); loop_dim++){
        for(int block_col = 0; block_col < (BLOCK_SIZE/TILE_SIZE); block_col++){
            for(int block_row = 0; block_row < (BLOCK_SIZE/TILE_SIZE); block_row++){
                int block_idx = block_col*(BLOCK_SIZE/TILE_SIZE)+block_row;
                for(int tile_col = 0; tile_col < (TILE_SIZE*TILE_SIZE/INPUT_PACK_SIZE); tile_col++){
                    int element_idx = block_idx*(TILE_SIZE*TILE_SIZE/INPUT_PACK_SIZE)+tile_col;
                    MemIntType row_b = input_B[element_idx];
                    MemVecType row_vec_b;
                    for(int i = 0; i < INPUT_PACK_SIZE; i++){
                        #pragma HLS UNROLL
                        row_vec_b[i] = row_b.range(8*i+7, 8*i);
                    }
                    stream_B.write(row_vec_b);
                }
            }
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
    MemStream& stream_A,
    MemStream& stream_B,
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C 
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
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> packed_A = stream_A.read();
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> packed_B = stream_B.read();
        
        for(int i = 0; i < TILE_SIZE; i++) {
            #pragma HLS UNROLL
            A_current[i] = packed_A[i];
            B_current[i] = packed_B[i];
        }

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
        hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> temp_data;
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            temp_data[j] = C_accum[i][j];
        }
        stream_C.write(temp_data);
    }
    
}

void base_mm(
    MemStream& stream_A,
    MemStream& stream_B,
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C 
){
    hls::vector<DTYPE_IN, INPUT_PACK_SIZE> A_buf[TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=1

    hls::vector<DTYPE_IN, INPUT_PACK_SIZE> B_buf[TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=2

    int32_t C_accum[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=C_accum complete dim=0

    // 初始化累加器
    for(int i = 0; i < TILE_SIZE; i++) {
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            C_accum[i][j] = 0;
        }
    }

    read_AB:
    for(int i = 0; i < TILE_SIZE; ++i){
        #pragma HLS PIPELINE
        A_buf[i] = stream_A.read();
        B_buf[i] = stream_B.read();
    }

    // 执行矩阵乘法
    row_for_AB:
    for(int i = 0; i < TILE_SIZE; ++i){
        col_for_AB:
        for(int j = 0; j < TILE_SIZE; ++j){
            int32_t sum = 0;
            product:
            for(int k = 0; k < TILE_SIZE; ++k){
                #pragma HLS UNROLL
                sum += A_buf[i][k] * B_buf[k][j];
            }
            C_accum[i][j] = sum;
        }
    }

    // 输出结果
    for(int i = 0; i < TILE_SIZE; ++i){
        hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> packed_C;
        for(int j = 0; j < TILE_SIZE; ++j){
            #pragma HLS UNROLL
            packed_C[j] = C_accum[i][j];
        }
        stream_C.write(packed_C);
    }
}

void base_mm_systolic(
    MemStream& stream_A,
    MemStream& stream_B,
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C 
){
    int8_t A_systolic[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=A_systolic complete dim=1
    #pragma HLS ARRAY_PARTITION variable=A_systolic complete dim=2
    
    int8_t B_systolic[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=B_systolic complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_systolic complete dim=2
    
    int32_t C_accum[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=C_accum block dim=1 factor=2

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
        #pragma HLS PIPELINE

        if(step < TILE_SIZE) {
            hls::vector<DTYPE_IN, INPUT_PACK_SIZE> packed_A = stream_A.read();
            hls::vector<DTYPE_IN, INPUT_PACK_SIZE> packed_B = stream_B.read();
            
            for(int i = 0; i < TILE_SIZE; i++) {
                #pragma HLS UNROLL
                A_systolic[0][i] = packed_A[i];
                B_systolic[i][0] = packed_B[i];
            }
        }
        
        // 数据传递
        // A矩阵向下传递
        for(int i = TILE_SIZE-1; i > 0; i--) {
            #pragma HLS UNROLL
            for(int j = 0; j < TILE_SIZE; j++) {
                #pragma HLS UNROLL
                A_systolic[i][j] = A_systolic[i-1][j];
            }
        }
        
        // B矩阵向右传递
        for(int j = TILE_SIZE-1; j > 0; j--) {
            #pragma HLS UNROLL
            for(int i = 0; i < TILE_SIZE; i++) {
                #pragma HLS UNROLL
                B_systolic[i][j] = B_systolic[i][j-1];
            }
        }
        
        for(int i = 0; i < TILE_SIZE; i++) {
            #pragma HLS UNROLL
            for(int j = 0; j < TILE_SIZE; j++) {
                #pragma HLS UNROLL
                if(i <= step && j <= step && i + j <= step) {
                    C_accum[i][j] += (int32_t)A_systolic[i][j] * B_systolic[i][j];
                }
            }
        }
    }
    
    for(int i = 0; i < TILE_SIZE; i++) {
        hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> temp_data;
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            temp_data[j] = C_accum[i][j];
        }
        stream_C << temp_data;
    }
}


void GemmBlock_wide(
    MemStream& stream_A,
    MemStream& stream_B,
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C
){
    #pragma HLS DATAFLOW

    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>> tile_C_stream;
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
                base_mm(stream_A, stream_B, tile_C_stream);
            }
        }
    }

    // 累加临时流
    // tile_C_stream: A[i][0] * B[0][j], A[i][1] * B[1][j], A[i][2] * B[2][j] ...

    accum_c_i:
    for(int block_i = 0; block_i < BLOCK_SIZE; block_i += TILE_SIZE) {
        #pragma HLS LOOP_TRIPCOUNT max=4 min=4 avg=4
        accum_c_j:
        for(int block_j = 0; block_j < BLOCK_SIZE; block_j += TILE_SIZE) {
            #pragma HLS PIPELINE
            #pragma HLS LOOP_TRIPCOUNT max=4 min=4 avg=4

            int32_t buffer_C[4][TILE_SIZE][TILE_SIZE];
            #pragma HLS ARRAY_PARTITION variable=buffer_C complete dim=0

            int32_t accum_C[TILE_SIZE][TILE_SIZE];
            #pragma HLS ARRAY_PARTITION variable=accum_C complete dim=0

            init_c_accum:
            for(int i = 0; i < TILE_SIZE; i++) {
                #pragma HLS UNROLL
                for(int j = 0; j < TILE_SIZE; j++) {
                    #pragma HLS UNROLL
                    accum_C[i][j] = 0;
                }
            }

            read_k:
            for(int block_k = 0; block_k < BLOCK_SIZE/TILE_SIZE; block_k++) {
                for(int tile_row = 0; tile_row < TILE_SIZE; tile_row++){
                    #pragma HLS PIPELINE
                    hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> pack_tile_c = tile_C_stream.read();
                    for(int j = 0; j < TILE_SIZE; j++){
                        #pragma HLS UNROLL
                        buffer_C[block_k][tile_row][j] = pack_tile_c[j];
                    }
                }

            }

            accum_k:
            for(int tile_row = 0; tile_row < TILE_SIZE; tile_row++){
                for(int j = 0; j < TILE_SIZE; j++){
                    #pragma HLS UNROLL
                    for(int block_k = 0; block_k < BLOCK_SIZE/TILE_SIZE; block_k++) {
                        #pragma HLS UNROLL
                        accum_C[tile_row][j] += buffer_C[block_k][tile_row][j];
                    }
                }
            }
            
            write_to_stream:
            for(int tile_row = 0; tile_row < TILE_SIZE; tile_row++) {
                #pragma HLS LOOP_TRIPCOUNT max=6 min=6 avg=6
                hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> temp_data;
                for(int j = 0; j < TILE_SIZE; j++) {
                    #pragma HLS UNROLL
                    temp_data[j] = accum_C[tile_row][j];
                }
                stream_C.write(temp_data);
            }
        }
    }

}


// 将打包的hls::vector输出流写入内存
void GemmWriteC_wide(
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C,
    ap_uint<192>* output_C
){
    #pragma HLS DATAFLOW

    const int total_blocks = (BLOCK_SIZE * BLOCK_SIZE) / 6;
    
    write_C:
    for(int block_i = 0; block_i < total_blocks; block_i++) {
        #pragma HLS PIPELINE II=1
        hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> packed_data = stream_C.read();
        ap_uint<192> output_data = 0;
        for(int i = 0; i < OUTPUT_PACK_SIZE; i++) {
            #pragma HLS UNROLL
            output_data.range(32*i+31, 32*i) = (ap_uint<32>)packed_data[i];
        }
        output_C[block_i] = output_data;
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

    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>> buf_A;
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>> buf_B;
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>> buf_C;

    #pragma HLS STREAM variable=buf_A depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS STREAM variable=buf_B depth=BLOCK_SIZE*BLOCK_SIZE
    #pragma HLS STREAM variable=buf_C depth=BLOCK_SIZE*BLOCK_SIZE

    GemmReadAB_wide(A, B, buf_A, buf_B);
    GemmBlock_wide(buf_A, buf_B, buf_C);
    GemmWriteC_wide(buf_C, C);

}
