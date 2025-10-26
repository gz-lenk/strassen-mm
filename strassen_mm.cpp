#include "strassen_mm.hpp"

void addTile_1(
    AXI_DATA_IN* buffer,
    unsigned int idx1,
    unsigned int sign1,
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_buf
){
    #pragma HLS INLINE

    for(int p = 0; p < TOTAL_TILE_ELEMENTS/INPUT_PACK_SIZE; p++){
        #pragma HLS PIPELINE
        AXI_DATA_IN packed_data_1 = buffer[idx1*TOTAL_TILE_ELEMENTS+p];
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> data;
        for (int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
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
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_buf
){
    #pragma HLS INLINE

    for(int p = 0; p < TOTAL_TILE_ELEMENTS/INPUT_PACK_SIZE; p++){
        #pragma HLS PIPELINE
        AXI_DATA_IN packed_data_1 = buffer[idx1*TOTAL_TILE_ELEMENTS+p];
        AXI_DATA_IN packed_data_2 = buffer[idx2*TOTAL_TILE_ELEMENTS+p];
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
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_buf
){
    #pragma HLS INLINE

    for(int p = 0; p < TOTAL_TILE_ELEMENTS/INPUT_PACK_SIZE; p++){
        #pragma HLS PIPELINE
        AXI_DATA_IN packed_data_1 = buffer[idx1*TOTAL_TILE_ELEMENTS+p];
        AXI_DATA_IN packed_data_2 = buffer[idx2*TOTAL_TILE_ELEMENTS+p];
        AXI_DATA_IN packed_data_3 = buffer[idx3*TOTAL_TILE_ELEMENTS+p];
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
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_buf
){
    #pragma HLS INLINE

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
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_A,
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_B
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

void base_mm(
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_A,
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_B,
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_M
){
    #pragma HLS DATAFLOW
    
    DTYPE_IN M_accum[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=M_accum complete dim=0

    init:
    for(int i = 0; i < TILE_SIZE; i++) {
        #pragma HLS PIPELINE
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            M_accum[i][j] = 0;
        }
    }

    mac:
    for(int k = 0; k < TILE_SIZE; k++) {
        #pragma HLS PIPELINE II=1
        
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> packed_A = stream_A.read();
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> packed_B = stream_B.read();

        for(int i = 0; i < TILE_SIZE; i++) {
            #pragma HLS UNROLL
            for(int j = 0; j < TILE_SIZE; j++) {
                #pragma HLS UNROLL
                M_accum[i][j] += (DTYPE_IN)packed_A[i] * packed_B[j];
            }
        }
    }

    shift:
    for(int i = 0; i < TILE_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> temp_data;
        for(int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS UNROLL
            temp_data[j] = M_accum[i][j];
        }
        stream_M << temp_data;
    }
}


void StrassenBlock(
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_A,
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_B,
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_M
){
    for(int idx = 0; idx < 49; idx++){
        base_mm(stream_A, stream_B, stream_M);
    }
}

void bufferTileStrassen_1(
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_M,
    unsigned int idx1,
    const bool sign1,
    DTYPE_OUT* buffer_c
){
    #pragma HLS INLINE

    for(int p = 0; p < TILE_SIZE*TILE_SIZE; p+=INPUT_PACK_SIZE){
        #pragma HLS PIPELINE
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> current_vec = stream_M.read();
        for(int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            if(sign1){
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
        }
    }
}

// stream_m : 49*Tile*Tile
// buffer_c : 16*Tile*Tile
void bufferTileStrassen_2(
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_M,
    unsigned int idx1,
    const bool sign1,
    unsigned int idx2,
    const bool sign2,
    DTYPE_OUT* buffer_c
){
    #pragma HLS INLINE

    for(int p = 0; p < TILE_SIZE*TILE_SIZE; p+=INPUT_PACK_SIZE){
        #pragma HLS PIPELINE
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> current_vec = stream_M.read();
        for(int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            if(sign1){
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
            if(sign2){
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
        }
    }
}

void bufferTileStrassen_3(
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_M,
    unsigned int idx1,
    const bool sign1,
    unsigned int idx2,
    const bool sign2,
    unsigned int idx3,
    const bool sign3,
    DTYPE_OUT* buffer_c
){
    #pragma HLS INLINE

    for(int p = 0; p < TILE_SIZE*TILE_SIZE; p+=INPUT_PACK_SIZE){
        #pragma HLS PIPELINE
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> current_vec = stream_M.read();
        for(int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            if(sign1){
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
            if(sign2){
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
            if(sign3){
                buffer_c[idx3*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx3*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
        }
    }
}

void bufferTileStrassen_4(
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_M,
    unsigned int idx1,
    const bool sign1,
    unsigned int idx2,
    const bool sign2,
    unsigned int idx3,
    const bool sign3,
    unsigned int idx4,
    const bool sign4,
    DTYPE_OUT* buffer_c
){
    #pragma HLS INLINE

    for(int p = 0; p < TILE_SIZE*TILE_SIZE; p+=INPUT_PACK_SIZE){
        #pragma HLS PIPELINE
        hls::vector<DTYPE_IN, INPUT_PACK_SIZE> current_vec = stream_M.read();
        for(int i = 0; i < INPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            if(sign1){
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx1*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
            if(sign2){
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx2*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
            if(sign3){
                buffer_c[idx3*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx3*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
            if(sign4){
                buffer_c[idx4*TILE_SIZE*TILE_SIZE+p+i] += (DTYPE_OUT)current_vec[i];
            }else{
                buffer_c[idx4*TILE_SIZE*TILE_SIZE+p+i] -= (DTYPE_OUT)current_vec[i];
            }
        }
    }
}



void StrassenOutBuffer(
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>>& stream_M,
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>>& stream_C
){
    #pragma HLS DATAFLOW

    DTYPE_OUT buffer_c[16*TILE_SIZE*TILE_SIZE];
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
        hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE> current_data;
        for(int i = 0; i < OUTPUT_PACK_SIZE; i++){
            #pragma HLS UNROLL
            current_data[i] = buffer_c[p+i];
        }
        stream_C << current_data;
    }
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
    AXI_DATA_OUT* C
){
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC depth=BLOCK_SIZE*BLOCK_SIZE/8

    #pragma HLS DATAFLOW

    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>> buf_A;
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>> buf_B;
    hls::stream<hls::vector<DTYPE_IN, INPUT_PACK_SIZE>> buf_M;
    hls::stream<hls::vector<DTYPE_OUT, OUTPUT_PACK_SIZE>> buf_C;

    #pragma HLS STREAM variable=buf_A depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS STREAM variable=buf_B depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS STREAM variable=buf_M depth=BLOCK_SIZE*BLOCK_SIZE/32
    #pragma HLS STREAM variable=buf_C depth=BLOCK_SIZE*BLOCK_SIZE/8

    StrassenReadAB(A, B, buf_A, buf_B);
    StrassenBlock(buf_A, buf_B, buf_M);
    StrassenOutBuffer(buf_M, buf_C);
    StrassenWriteC(buf_C, C);

}