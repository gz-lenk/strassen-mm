#include "ap_int.h"
#include "hls_stream.h"
#include "strassen_mm.hpp"
#include <stdio.h>
#include <iostream>
#include <cstdlib>

void generate_random_matrix(DTYPE_IN* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 256 - 128; // -128 到 127
    }
}

void rearrange_and_pack_A(DTYPE_IN* A_original, AXI_DATA_IN* A_packed) {
    int packed_index = 0;

    for (int block_i = 0; block_i < BLOCK_SIZE; block_i += TILE_SIZE) {
        for (int block_j = 0; block_j < BLOCK_SIZE; block_j += TILE_SIZE) {
            for (int tile_row = 0; tile_row < TILE_SIZE; tile_row++) {
                AXI_DATA_IN packed_value = 0;
                for (int tile_col = 0; tile_col < TILE_SIZE; tile_col++) {
                    int original_row = block_i + tile_row;
                    int original_col = block_j + tile_col;
                    int8_t value = A_original[original_row * BLOCK_SIZE + original_col];
                    packed_value.range(8 * tile_col + 7, 8 * tile_col) = (ap_uint<8>)value;
                }
                A_packed[packed_index++] = packed_value;
            }
        }
    }   
}

void rearrange_and_pack_B(DTYPE_IN* B_original, AXI_DATA_IN* B_packed) {
    int packed_index = 0;
    
    for (int block_j = 0; block_j < BLOCK_SIZE; block_j += TILE_SIZE) {
        for (int block_i = 0; block_i < BLOCK_SIZE; block_i += TILE_SIZE) {
            for (int tile_col = 0; tile_col < TILE_SIZE; tile_col++) {
                AXI_DATA_IN packed_value = 0;
                for (int tile_row = 0; tile_row < TILE_SIZE; tile_row++) {
                    int original_row = block_i + tile_row;
                    int original_col = block_j + tile_col;
                    int8_t value = B_original[original_row * BLOCK_SIZE + original_col];
                    packed_value.range(8 * tile_row + 7, 8 * tile_row) = (ap_uint<8>)value;
                }
                B_packed[packed_index++] = packed_value;
            }
        }
    }
}

void software_matmul(int8_t* A, int8_t* B, int32_t* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int32_t sum = 0;
            for (int k = 0; k < size; k++) {
                sum += (int32_t)A[i * size + k] * (int32_t)B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

void unpack_C(ap_uint<256>* C_packed, int8_t* C_unpacked) {
    int unpacked_index = 0;
    
    for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / INPUT_PACK_SIZE; i++) {
        ap_uint<256> packed_data = C_packed[i];
        for (int j = 0; j < INPUT_PACK_SIZE; j++) {
            C_unpacked[unpacked_index++] = (int8_t)packed_data.range(8 * j + 7, 8 * j);
        }
    }
}

void rearrange_C(int8_t* C_hardware, int8_t* C_normal) {
    // 硬件输出的顺序：按block_i, block_j, tile_row, tile_col
    // 每个tile输出64行，每行包含64个元素
    int hardware_index = 0;

    for (int block_i = 0; block_i < BLOCK_SIZE; block_i += TILE_SIZE) {
        for (int block_j = 0; block_j < BLOCK_SIZE; block_j += TILE_SIZE) {
            for (int tile_row = 0; tile_row < TILE_SIZE; tile_row++) {
                // 从硬件输出中读取一个packed数据（包含64个元素）
                int8_t temp_data[TILE_SIZE];
                for (int j = 0; j < TILE_SIZE; j++) {
                    temp_data[j] = C_hardware[hardware_index * TILE_SIZE + j];
                }
                
                // 将64个元素分配到对应的位置
                for (int tile_col = 0; tile_col < TILE_SIZE; tile_col++) {
                    int normal_index = (block_i + tile_row) * BLOCK_SIZE + (block_j + tile_col);
                    C_normal[normal_index] = temp_data[tile_col];
                }
                
                hardware_index++;
            }
        }
    }
}


bool compare_matrices(int8_t* C_hw, int8_t* C_sw, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (C_hw[i * size + j] != C_sw[i * size + j]) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "HW = " << C_hw[i * size + j] 
                          << ", SW = " << C_sw[i * size + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}


int test_mm_pipeline(){
    std::cout << "\n=== Testing mm_pipeline ===" << std::endl;
    DTYPE_IN A_original[BLOCK_SIZE * BLOCK_SIZE];
    DTYPE_IN B_original[BLOCK_SIZE * BLOCK_SIZE];
    DTYPE_IN C_original[BLOCK_SIZE * BLOCK_SIZE];

    const int packed_A_size = (BLOCK_SIZE * BLOCK_SIZE) / 32;  // 每个 ap_uint<256> 包含 32 个元素
    const int packed_B_size = (BLOCK_SIZE * BLOCK_SIZE) / 32;  // 每个 ap_uint<256> 包含 32 个元素

    AXI_DATA_IN A_packed[packed_A_size];
    AXI_DATA_IN B_packed[packed_B_size];
    AXI_DATA_IN C_packed[packed_A_size];

    DTYPE_IN C_hw[BLOCK_SIZE * BLOCK_SIZE];

    srand(42);

    generate_random_matrix(A_original, BLOCK_SIZE * BLOCK_SIZE);
    generate_random_matrix(B_original, BLOCK_SIZE * BLOCK_SIZE);

    rearrange_and_pack_A(A_original, A_packed);
    rearrange_and_pack_B(B_original, B_packed);

    std::cout << "Calling mm_pipeline..." << std::endl;

    mm_pipeline(A_packed, B_packed, C_packed);

    unpack_C(C_packed, C_original);
    rearrange_C(C_original, C_hw);

    bool success = compare_matrices(C_hw, A_original, BLOCK_SIZE);

    if (success) {
        std::cout << "mm_pipeline test PASSED!" << std::endl;
    } else {
        std::cout << "mm_pipeline test FAILED!" << std::endl;
        
        std::cout << "First 10 elements comparison:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "C_hw[" << i << "] = " << C_hw[i] 
                      << ", A_original[" << i << "] = " << A_original[i] << std::endl;
        }
    }

    return success ? 0 : 1;

}

int main() {
    std::cout << "Starting mm_pipeline Testbench" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int error_count = 0;
    
    // 运行所有测试
    error_count += test_mm_pipeline();
    
    std::cout << "\n========================================" << std::endl;
    if (error_count == 0) {
        std::cout << "ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << error_count << " test(s) FAILED!" << std::endl;
    }
    
    return error_count;
}
