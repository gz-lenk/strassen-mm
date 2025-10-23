#include "strassen_mm.hpp"

void addTile_1(
    AXI_DATA_IN* buffer,
    unsigned int idx1,
    unsigned int sign1,
    hls::stream<hls::vector<DTYPE_IN, 32>>& stream_buf
){
    for(int p = 0; p < TOTAL_TILE_ELEMENTS/INPUT_PACK_SIZE; p++){
        AXI_DATA_IN packed_data_1 = buffer[idx1*TOTAL_TILE_ELEMENTS+p];
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> data;
        for (int i = 0; i < INPUT_PACK_SIZE; i++){
            if (sign1) {
                data[i] += packed_data_1(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_1(8*(i+1)-1, 8*i);
            }
        }
        stream_buf << data;
    }
}

void addTile_2(
    AXI_DATA_IN* buffer,
    unsigned int idx1,
    unsigned int idx2,
    unsigned int sign1,
    unsigned int sign2,
    hls::stream<hls::vector<DTYPE_IN, 32>>& stream_buf
){
    for(int p = 0; p < TOTAL_TILE_ELEMENTS/INPUT_PACK_SIZE; p++){
        AXI_DATA_IN packed_data_1 = buffer[idx1*TOTAL_TILE_ELEMENTS+p];
        AXI_DATA_IN packed_data_2 = buffer[idx2*TOTAL_TILE_ELEMENTS+p];
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> data;
        for (int i = 0; i < INPUT_PACK_SIZE; i++){
            if (sign1) {
                data[i] += packed_data_1(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_1(8*(i+1)-1, 8*i);
            }
            if (sign2) {
                data[i] += packed_data_2(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_2(8*(i+1)-1, 8*i);
            }
        }
        stream_buf << data;
    }
}

void addTile_3(
    AXI_DATA_IN* buffer,
    unsigned int idx1,
    unsigned int idx2,
    unsigned int idx3,
    unsigned int sign1,
    unsigned int sign2,
    unsigned int sign3,
    hls::stream<hls::vector<DTYPE_IN, 32>>& stream_buf
){
    for(int p = 0; p < TOTAL_TILE_ELEMENTS/INPUT_PACK_SIZE; p++){
        AXI_DATA_IN packed_data_1 = buffer[idx1*TOTAL_TILE_ELEMENTS+p];
        AXI_DATA_IN packed_data_2 = buffer[idx2*TOTAL_TILE_ELEMENTS+p];
        AXI_DATA_IN packed_data_3 = buffer[idx3*TOTAL_TILE_ELEMENTS+p];
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> data;
        for (int i = 0; i < INPUT_PACK_SIZE; i++){
            if (sign1) {
                data[i] += packed_data_1(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_1(8*(i+1)-1, 8*i);
            }
            if (sign2) {
                data[i] += packed_data_2(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_2(8*(i+1)-1, 8*i);
            }
            if (sign3) {
                data[i] += packed_data_3(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_3(8*(i+1)-1, 8*i);
            }
        }
        stream_buf << data;
    }
}

void addTile_4(
    AXI_DATA_IN* buffer,
    unsigned int idx1,
    unsigned int idx2,
    unsigned int idx3,
    unsigned int idx4,
    unsigned int sign1,
    unsigned int sign2,
    unsigned int sign3,
    unsigned int sign4,
    hls::stream<hls::vector<DTYPE_IN, 32>>& stream_buf
){
    for(int p = 0; p < TOTAL_TILE_ELEMENTS/INPUT_PACK_SIZE; p++){
        #pragma HLS PIPELINE
        AXI_DATA_IN packed_data_1 = buffer[idx1*TOTAL_TILE_ELEMENTS+p];
        AXI_DATA_IN packed_data_2 = buffer[idx2*TOTAL_TILE_ELEMENTS+p];
        AXI_DATA_IN packed_data_3 = buffer[idx3*TOTAL_TILE_ELEMENTS+p];
        AXI_DATA_IN packed_data_4 = buffer[idx4*TOTAL_TILE_ELEMENTS+p];
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> data;
        for (int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            if (sign1) {
                data[i] += packed_data_1(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_1(8*(i+1)-1, 8*i);
            }
            if (sign2) {
                data[i] += packed_data_2(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_2(8*(i+1)-1, 8*i);
            }
            if (sign3) {
                data[i] += packed_data_3(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_3(8*(i+1)-1, 8*i);
            }
            if (sign4) {
                data[i] += packed_data_4(8*(i+1)-1, 8*i);
            } else {
                data[i] -= packed_data_4(8*(i+1)-1, 8*i);
            }
        }
        stream_buf << data;
    }
}


void StrassenReadAB(
    AXI_DATA_IN* input_A,
    AXI_DATA_IN* input_B,
    hls::stream<hls::vector<DTYPE_IN, 32>>& stream_A,
    hls::stream<hls::vector<DTYPE_IN, 32>>& stream_B
){
    #pragma HLS DATAFLOW
    // read_A:
    // for(int p = 0; p < TOTAL_BLOCK_ELEMENTS/INPUT_PACK_SIZE; p++){
    //     #pragma HLS PIPELINE
    //     AXI_DATA_IN packed_data = input_A[p];
    //     hls::vector<DTYPE_IN, INPUT_PACK_SIZE> data;
    //     read_package_a:
    //     for (int i = 0; i < INPUT_PACK_SIZE; i++) {
    //         #pragma HLS UNROLL
    //         data[i] = static_cast<DTYPE_IN>(packed_data(8*(i+1)-1, 8*i));
    //     }
    //     stream_A << data;
    // }

    // read_B:
    // for(int p = 0; p < TOTAL_BLOCK_ELEMENTS/INPUT_PACK_SIZE; p++){
    //     #pragma HLS PIPELINE
    //     AXI_DATA_IN packed_data = input_B[p];
    //     hls::vector<DTYPE_IN, INPUT_PACK_SIZE> data;
    //     read_package_b:
    //     for (int i = 0; i < INPUT_PACK_SIZE; i++) {
    //         #pragma HLS UNROLL
    //         data[i] = static_cast<DTYPE_IN>(packed_data(8*(i+1)-1, 8*i));
    //     }
    //     stream_B << data;
    // }

    ///////////////////////////////////////////////////////////////////
    // code for A to A_linear
    ///////////////////////////////////////////////////////////////////
    addTile_4(input_A, 0, 3, 12, 15, 1, 1, 1, 1, stream_A);
    addTile_2(input_A, 0, 12, 1, 1, stream_A);
    addTile_4(input_A, 1, 3, 13, 15, 1, 0, 1, 0, stream_A);
    addTile_4(input_A, 0, 1, 12, 13, 0, 1, 0, 1, stream_A);
    addTile_2(input_A, 2, 14, 1, 1, stream_A);
    addTile_4(input_A, 0, 1, 12, 13, 1, 1, 1, 1, stream_A);
    addTile_4(input_A, 2, 3, 14, 15, 1, 1, 1, 1, stream_A);
    addTile_2(input_A, 0, 3, 1, 1, stream_A);
    addTile_1(input_A, 0, 1, stream_A);
    addTile_2(input_A, 1, 3, 1, 0, stream_A);
    addTile_2(input_A, 0, 1, 0, 1, stream_A);
    addTile_1(input_A, 2, 1, stream_A);
    addTile_2(input_A, 0, 1, 1, 1, stream_A);
    addTile_2(input_A, 2, 3, 1, 1, stream_A);
    addTile_4(input_A, 4, 7, 12, 15, 1, 1, 0, 0, stream_A);
    addTile_2(input_A, 4, 12, 1, 0, stream_A);
    addTile_4(input_A, 5, 7, 13, 15, 1, 0, 0, 1, stream_A);
    addTile_4(input_A, 4, 5, 12, 13, 0, 1, 1, 0, stream_A);
    addTile_2(input_A, 6, 14, 1, 0, stream_A);
    addTile_4(input_A, 4, 5, 12, 13, 1, 1, 0, 0, stream_A);
    addTile_4(input_A, 6, 7, 14, 15, 1, 1, 0, 0, stream_A);
    addTile_4(input_A, 0, 3, 4, 7, 0, 0, 1, 1, stream_A);
    addTile_2(input_A, 0, 4, 0, 1, stream_A);
    addTile_4(input_A, 1, 3, 5, 7, 0, 1, 1, 0, stream_A);
    addTile_4(input_A, 0, 1, 4, 5, 1, 0, 0, 1, stream_A);
    addTile_2(input_A, 2, 6, 0, 1, stream_A);
    addTile_4(input_A, 0, 1, 4, 5, 0, 0, 1, 1, stream_A);
    addTile_4(input_A, 2, 3, 6, 7, 0, 0, 1, 1, stream_A);
    addTile_2(input_A, 8, 11, 1, 1, stream_A);
    addTile_1(input_A, 8, 1, stream_A);
    addTile_2(input_A, 9, 11, 1, 0, stream_A);
    addTile_2(input_A, 8, 9, 0, 1, stream_A);
    addTile_1(input_A, 10, 1, stream_A);
    addTile_2(input_A, 8, 9, 1, 1, stream_A);
    addTile_2(input_A, 10, 11, 1, 1, stream_A);
    addTile_4(input_A, 0, 3, 4, 7, 1, 1, 1, 1, stream_A);
    addTile_2(input_A, 0, 4, 1, 1, stream_A);
    addTile_4(input_A, 1, 3, 5, 7, 1, 0, 1, 0, stream_A);
    addTile_4(input_A, 0, 1, 4, 5, 0, 1, 0, 1, stream_A);
    addTile_2(input_A, 2, 6, 1, 1, stream_A);
    addTile_4(input_A, 0, 1, 4, 5, 1, 1, 1, 1, stream_A);
    addTile_4(input_A, 2, 3, 6, 7, 1, 1, 1, 1, stream_A);
    addTile_4(input_A, 8, 11, 12, 15, 1, 1, 1, 1, stream_A);
    addTile_2(input_A, 8, 12, 1, 1, stream_A);
    addTile_4(input_A, 9, 11, 13, 15, 1, 0, 1, 0, stream_A);
    addTile_4(input_A, 8, 9, 12, 13, 0, 1, 0, 1, stream_A);
    addTile_2(input_A, 10, 14, 1, 1, stream_A);
    addTile_4(input_A, 8, 9, 12, 13, 1, 1, 1, 1, stream_A);
    addTile_4(input_A, 10, 11, 14, 15, 1, 1, 1, 1, stream_A);
    ///////////////////////////////////////////////////////////////////
    // code for B to B_linear
    ///////////////////////////////////////////////////////////////////
    addTile_4(input_B, 0, 1, 4, 5, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 1, 2, 5, 6, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 0, 3, 4, 7, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 2, 3, 6, 7, 1, 0, 1, 0, stream_B);
    addTile_4(input_B, 0, 1, 4, 5, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 1, 2, 5, 6, 0, 0, 0, 0, stream_B);
    addTile_4(input_B, 0, 3, 4, 7, 0, 1, 0, 1, stream_B);
    addTile_4(input_B, 4, 5, 8, 9, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 5, 6, 9, 10, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 4, 7, 8, 11, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 6, 7, 10, 11, 1, 0, 1, 0, stream_B);
    addTile_4(input_B, 4, 5, 8, 9, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 5, 6, 9, 10, 0, 0, 0, 0, stream_B);
    addTile_4(input_B, 4, 7, 8, 11, 0, 1, 0, 1, stream_B);
    addTile_4(input_B, 0, 1, 12, 13, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 1, 2, 13, 14, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 0, 3, 12, 15, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 2, 3, 14, 15, 1, 0, 1, 0, stream_B);
    addTile_4(input_B, 0, 1, 12, 13, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 1, 2, 13, 14, 0, 0, 0, 0, stream_B);
    addTile_4(input_B, 0, 3, 12, 15, 0, 1, 0, 1, stream_B);
    addTile_4(input_B, 8, 9, 12, 13, 1, 1, 0, 0, stream_B);
    addTile_4(input_B, 9, 10, 13, 14, 1, 1, 0, 0, stream_B);
    addTile_4(input_B, 8, 11, 12, 15, 1, 1, 0, 0, stream_B);
    addTile_4(input_B, 10, 11, 14, 15, 1, 0, 0, 1, stream_B);
    addTile_4(input_B, 8, 9, 12, 13, 1, 1, 0, 0, stream_B);
    addTile_4(input_B, 9, 10, 13, 14, 0, 0, 1, 1, stream_B);
    addTile_4(input_B, 8, 11, 12, 15, 0, 1, 1, 0, stream_B);
    addTile_4(input_B, 0, 1, 4, 5, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 1, 2, 5, 6, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 0, 3, 4, 7, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 2, 3, 6, 7, 1, 0, 1, 0, stream_B);
    addTile_4(input_B, 0, 1, 4, 5, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 1, 2, 5, 6, 0, 0, 0, 0, stream_B);
    addTile_4(input_B, 0, 3, 4, 7, 0, 1, 0, 1, stream_B);
    addTile_4(input_B, 4, 5, 8, 9, 0, 0, 0, 0, stream_B);
    addTile_4(input_B, 5, 6, 9, 10, 0, 0, 0, 0, stream_B);
    addTile_4(input_B, 4, 7, 8, 11, 0, 0, 0, 0, stream_B);
    addTile_4(input_B, 6, 7, 10, 11, 0, 1, 0, 1, stream_B);
    addTile_4(input_B, 4, 5, 8, 9, 0, 0, 0, 0, stream_B);
    addTile_4(input_B, 5, 6, 9, 10, 1, 1, 1, 1, stream_B);
    addTile_4(input_B, 4, 7, 8, 11, 1, 0, 1, 0, stream_B);
    addTile_4(input_B, 0, 1, 12, 13, 0, 0, 1, 1, stream_B);
    addTile_4(input_B, 1, 2, 13, 14, 0, 0, 1, 1, stream_B);
    addTile_4(input_B, 0, 3, 12, 15, 0, 0, 1, 1, stream_B);
    addTile_4(input_B, 2, 3, 14, 15, 0, 1, 1, 0, stream_B);
    addTile_4(input_B, 0, 1, 12, 13, 0, 0, 1, 1, stream_B);
    addTile_4(input_B, 1, 2, 13, 14, 1, 1, 0, 0, stream_B);
    addTile_4(input_B, 0, 3, 12, 15, 1, 0, 0, 1, stream_B);
    
}

void StrassenWriteC(
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C,
    AXI_DATA_OUT* output_C
){
    for(int p = 0; p < TOTAL_BLOCK_ELEMENTS/OUTPUT_PACK_SIZE; p++){
        #pragma HLS PIPELINE
        AXI_DATA_OUT packed_data = 0;
        hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> current_vec = stream_C.read();
        write_package_c:
        for (int i = 0; i < OUTPUT_PACK_SIZE; i++) {
            #pragma HLS UNROLL
            packed_data(32*(i+1)-1, 32*i) = static_cast<DTYPE_OUT>(current_vec[i]);
        }
        output_C[p] = packed_data;
    }
}

void mm_pipeline(
    AXI_DATA_IN* A,
    AXI_DATA_IN* B,
    AXI_DATA_IN* C
){
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC depth=BLOCK_SIZE*BLOCK_SIZE/8

    #pragma HLS DATAFLOW

    hls::stream<hls::vector<DTYPE_IN, 32>> buf_A;
    hls::stream<hls::vector<DTYPE_IN, 32>> buf_B;
    hls::stream<hls::vector<DTYPE_IN, 32>> buf_M;
    hls::stream<hls::vector<DTYPE_OUT, 8>> buf_C;

    #pragma HLS STREAM variable=buf_A depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS STREAM variable=buf_B depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS STREAM variable=buf_C depth=BLOCK_SIZE*BLOCK_SIZE/8

    StrassenReadAB(A, B, buf_A, buf_B);
    // StrassenBlock(buf_A, buf_B, buf_M);
    // StrassenOutBuffer(buf_M, buf_C);
    // StrassenWriteC(buf_C, C);

}