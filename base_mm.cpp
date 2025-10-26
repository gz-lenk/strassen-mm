#include "ap_int.h"
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>

// typedef ap_uint<192>            MemIntType;
// typedef hls::stream<MemIntType> MemStream;

#define PACK_SIZE           32             
#define PACKED_C            8              

#define MAT_SIZE            768
#define BLOCK_SIZE          24
#define TILE_SIZE           6       

#define PARALLEL_M          6
#define PARALLEL_N          6

void GemmReadAB(
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

void base_mm(
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B,
    int32_t* buffer_C
){

    int8_t A_current[TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=A_current complete
    int8_t B_current[TILE_SIZE];  
    #pragma HLS ARRAY_PARTITION variable=B_current complete
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
                buffer_C[i*TILE_SIZE+j] += (int32_t)A_current[i] * B_current[j];
            }
        }
    }
}

void stream_buffer(
    int32_t* output_C, 
    hls::stream<ap_uint<192>>& stream_C 
){
    out_buffer:
    for(int i = 0; i < TILE_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        int32_t temp_data[TILE_SIZE];
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            temp_data[j] = output_C[i*TILE_SIZE+j];
        }
        ap_uint<192> packed_C = pack_int32_to_ap_uint192(temp_data);
        stream_C.write(packed_C);
    }
}

// void GemmBlock(
//     hls::stream<ap_uint<48>>& stream_A,
//     hls::stream<ap_uint<48>>& stream_B,
//     hls::stream<ap_uint<192>>& stream_C 
// ){
//     #pragma HLS DATAFLOW
//     int32_t output_C[BLOCK_SIZE/TILE_SIZE][BLOCK_SIZE/TILE_SIZE][TILE_SIZE][TILE_SIZE];
//     #pragma HLS ARRAY_PARTITION variable=output_C complete dim=3
//     #pragma HLS ARRAY_PARTITION variable=output_C complete dim=4

//     init:
//     for(int i = 0; i < BLOCK_SIZE/TILE_SIZE; i++){
//         for(int j = 0; j < BLOCK_SIZE/TILE_SIZE; j++){
//             for(int x = 0; x < TILE_SIZE; x++){
//                 #pragma HLS UNROLL
//                 for(int y = 0; y < TILE_SIZE; y++){
//                     #pragma HLS UNROLL
//                     output_C[i][j][x][y] = 0;
//                 }
//             }
//         }
//     }


//     i_block:
//     for(int block_i = 0; block_i < BLOCK_SIZE; block_i+=TILE_SIZE){
//         j_block:
//         for(int block_j = 0; block_j < BLOCK_SIZE; block_j+=TILE_SIZE){
//             k_block:
//             for(int block_k = 0; block_k < BLOCK_SIZE; block_k+=TILE_SIZE){
//                 #pragma HLS PIPELINE
//                 // 子块乘法：C[i][j] += A[i][k] × B[k][j]
//                 int32_t buffer_C[TILE_SIZE][TILE_SIZE];
//                 #pragma HLS ARRAY_PARTITION variable=buffer_C complete dim=1
//                 #pragma HLS ARRAY_PARTITION variable=buffer_C complete dim=2

//                 base_mm(stream_A, stream_B, (int32_t*)buffer_C);
//                 accum_mat:
//                 for(int x = 0; x < TILE_SIZE; x++) {
//                     #pragma HLS UNROLL
//                     for(int y = 0; y < TILE_SIZE; y++) {
//                         #pragma HLS UNROLL
//                         output_C[block_i / TILE_SIZE][block_j / TILE_SIZE][x][y] += buffer_C[x][y];
//                     }
//                 }
//             }
//             stream_buffer_i_j:
//             for(int x = 0; x < TILE_SIZE; x++) {
//                 #pragma HLS PIPELINE II=1
//                 int32_t temp_data[TILE_SIZE];
//                 for(int y = 0; y < TILE_SIZE; y++) {
//                     #pragma HLS UNROLL
//                     temp_data[y] = output_C[block_i / TILE_SIZE][block_j / TILE_SIZE][x][y];
//                 }
//                 ap_uint<192> packed_C = pack_int32_to_ap_uint192(temp_data);
//                 stream_C.write(packed_C);
//             }
//         }
//     }

// }

void accum_mat(
    int32_t* buffer_C,
    int32_t* output_C
){
    accum_mat:
    for(int i = 0; i < TILE_SIZE; i++) {
        #pragma HLS UNROLL
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            output_C[i*TILE_SIZE+j] += buffer_C[i*TILE_SIZE+j];
        }
    }
}


// 核心修改：GemmBlock中K循环外提
void GemmBlock(
    hls::stream<ap_uint<48>>& stream_A,
    hls::stream<ap_uint<48>>& stream_B,
    hls::stream<ap_uint<192>>& stream_C 
){
    // 为每个(i,j)块分配独立的输出缓冲区（避免跨块冲突）
    int32_t output_C[BLOCK_SIZE/TILE_SIZE][BLOCK_SIZE/TILE_SIZE][TILE_SIZE*TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=output_C complete dim=3  // 展开内部维度加速访问
    int32_t buffer_C[TILE_SIZE*TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=buffer_C complete

    // 初始化所有(i,j)块的输出缓冲区为0
    init_output:
    for(int i_idx = 0; i_idx < BLOCK_SIZE/TILE_SIZE; i_idx++) {
        for(int j_idx = 0; j_idx < BLOCK_SIZE/TILE_SIZE; j_idx++) {
            for(int idx = 0; idx < TILE_SIZE*TILE_SIZE; idx++) {
                #pragma HLS UNROLL
                output_C[i_idx][j_idx][idx] = 0;
            }
        }
    }

    // 1. K维度循环外提（最外层）：按K块迭代累加
    k_block:
    for(int block_k = 0; block_k < BLOCK_SIZE; block_k += TILE_SIZE) {
        // 2. 内层循环：遍历所有(i,j)块，用当前K块的数据更新
        i_block:
        for(int block_i = 0; block_i < BLOCK_SIZE; block_i += TILE_SIZE) {
            int i_idx = block_i / TILE_SIZE;  // (i,j)块索引
            j_block:
            for(int block_j = 0; block_j < BLOCK_SIZE; block_j += TILE_SIZE) {
                #pragma HLS PIPELINE II=1  // 流水线作用于(i,j)块，无跨K依赖
                int j_idx = block_j / TILE_SIZE;

                // 重置buffer_C（当前K迭代的临时结果）
                reset_buffer:
                for(int idx = 0; idx < TILE_SIZE*TILE_SIZE; idx++) {
                    #pragma HLS UNROLL
                    buffer_C[idx] = 0;
                }

                // 计算当前(i,k,j)块的乘积：A[i][k] × B[k][j]
                base_mm(stream_A, stream_B, buffer_C);

                // 累加至(i,j)块的输出结果（仅当前K迭代，无跨迭代冲突）
                accum_mat(buffer_C, output_C[i_idx][j_idx]);
            }
        }
    }

    // 3. 所有K迭代完成后，输出所有(i,j)块的最终结果
    output_all:
    for(int i_idx = 0; i_idx < BLOCK_SIZE/TILE_SIZE; i_idx++) {
        for(int j_idx = 0; j_idx < BLOCK_SIZE/TILE_SIZE; j_idx++) {
            #pragma HLS PIPELINE II=1
            stream_buffer(output_C[i_idx][j_idx], stream_C);
        }
    }
}

// 将打包的ap_uint<192>输出流写入内存
void GemmWriteC(
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

    GemmReadAB(A, B, buf_A, buf_B);
    GemmBlock(buf_A, buf_B, buf_C);
    GemmWriteC(buf_C, C);

}
