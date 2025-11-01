#include "ap_int.h"
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

void mm_pipeline(
    ap_uint<192>* A,
    ap_uint<192>* B,
    ap_uint<192>* C
);

#define TILE_SIZE 6
#define BLOCK_SIZE 24

#define INPUT_PACK_SIZE     24
#define OUTPUT_PACK_SIZE    6

// 生成随机矩阵
void generate_random_matrix(int8_t* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 256 - 128; // -128 到 127
    }
}

void pack_A(int8_t* A_original, ap_uint<192>* A_packed){
    int packed_idx = 0;
    for (int block_row = 0; block_row < BLOCK_SIZE; block_row += TILE_SIZE){
        for(int tile_row = 0; tile_row < TILE_SIZE; tile_row++){
            ap_uint<192> packed_value = 0;
            for(int i = 0; i < INPUT_PACK_SIZE; i++){
                int original_row = block_row + tile_row;
                int original_col = i;
                int8_t value = A_original[original_row * BLOCK_SIZE + original_col];
                packed_value.range(8 * i + 7, 8 * i) = (ap_uint<8>)value;
            }
            A_packed[packed_idx++] = packed_value;
        }
    }
}

// 打包B矩阵
void pack_B(int8_t* B_original, ap_uint<192>* B_packed) {
    int packed_index = 0;
    for (int block_col = 0; block_col < BLOCK_SIZE; block_col += TILE_SIZE) {
        for (int tile_col = 0; tile_col < TILE_SIZE; tile_col++) {
            ap_uint<192> packed_value = 0;
            for (int i = 0; i < INPUT_PACK_SIZE; i++) {
                int original_row = i;
                int original_col = block_col + tile_col;
                int8_t value = B_original[original_row * BLOCK_SIZE + original_col];
                packed_value.range(8 * i + 7, 8 * i) = (ap_uint<8>)value;
            }
            B_packed[packed_index++] = packed_value;
        }
    }
}

// 解包C矩阵
void unpack_C(ap_uint<192>* C_packed, int32_t* C_unpacked) {
    int unpacked_index = 0;
    for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / OUTPUT_PACK_SIZE; i++) {
        ap_uint<192> packed_data = C_packed[i];
        for (int j = 0; j < 6; j++) {
            if (unpacked_index < BLOCK_SIZE * BLOCK_SIZE) {
                C_unpacked[unpacked_index++] = (int32_t)packed_data.range(32 * j + 31, 32 * j);
            }
        }
    }
}

// 重排C矩阵到正常顺序
void rearrange_C(int32_t* C_hardware, int32_t* C_normal) {
    int hardware_index = 0;
    for(int block_col = 0; block_col < BLOCK_SIZE; block_col += TILE_SIZE){
        for(int block_row = 0; block_row < BLOCK_SIZE; block_row ++){
            int32_t temp_data[TILE_SIZE];
            for(int j = 0; j < TILE_SIZE; j++){
                temp_data[j] = C_hardware[hardware_index * TILE_SIZE + j];
                int normal_index = block_row * BLOCK_SIZE + block_col + j;
                C_normal[normal_index] = temp_data[j];
            }
            hardware_index++;
        }
    }
}

// 软件矩阵乘法参考实现
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

// 比较矩阵
bool compare_matrices(int32_t* C_hw, int32_t* C_sw, int size) {
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

// 测试随机矩阵
int test_random_matrices() {
    std::cout << "=== Testing Random Matrices ===" << std::endl;

    const int matrix_size = BLOCK_SIZE * BLOCK_SIZE;
    const int packed_size_input = matrix_size / 24;
    const int packed_size_output = matrix_size / 6;
    
    int8_t A_original[matrix_size];
    int8_t B_original[matrix_size];
    ap_uint<192> A_packed[packed_size_input];
    ap_uint<192> B_packed[packed_size_input];
    ap_uint<192> C_packed[packed_size_output];
    int32_t C_hw[matrix_size];
    int32_t C_sw[matrix_size];

    srand(42);
    generate_random_matrix(A_original, matrix_size);
    generate_random_matrix(B_original, matrix_size);

    pack_A(A_original, A_packed);
    pack_B(B_original, B_packed);

    std::cout << "Calling mm_pipeline..." << std::endl;
    mm_pipeline(A_packed, B_packed, C_packed);

    unpack_C(C_packed, C_hw);
    rearrange_C(C_hw, C_hw);
    software_matmul(A_original, B_original, C_sw, BLOCK_SIZE);

    bool success = compare_matrices(C_hw, C_sw, BLOCK_SIZE);

    if (success) {
        std::cout << "Random matrices test PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Random matrices test FAILED!" << std::endl;
        std::cout << "First 10 elements comparison:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "C_hw[" << i << "] = " << C_hw[i] 
                      << ", C_sw[" << i << "] = " << C_sw[i] << std::endl;
        }
        return 1;
    }
}

// 测试边界情况
int test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;
    int error_count = 0;
    
    const int matrix_size = BLOCK_SIZE * BLOCK_SIZE;
    const int packed_size = matrix_size / 6;
    
    // 测试1: 全零矩阵
    std::cout << "Test 1: Zero matrices" << std::endl;
    {
        int8_t A_original[matrix_size] = {0};
        int8_t B_original[matrix_size] = {0};
        ap_uint<192> A_packed[24];
        ap_uint<192> B_packed[24];
        ap_uint<192> C_packed[packed_size];
        int32_t C_hw[matrix_size];
        int32_t C_sw[matrix_size];
        
        pack_A(A_original, A_packed);
        pack_B(B_original, B_packed);
        
        mm_pipeline(A_packed, B_packed, C_packed);
        unpack_C(C_packed, C_hw);
        rearrange_C(C_hw, C_hw);
        software_matmul(A_original, B_original, C_sw, BLOCK_SIZE);
        
        if (!compare_matrices(C_hw, C_sw, BLOCK_SIZE)) {
            std::cout << "Zero matrices test FAILED!" << std::endl;
            error_count++;
        } else {
            std::cout << "Zero matrices test PASSED!" << std::endl;
        }
    }
    
    // 测试2: 单位矩阵
    std::cout << "Test 2: Identity matrix" << std::endl;
    {
        int8_t A_original[matrix_size];
        int8_t B_original[matrix_size];
        ap_uint<192> A_packed[24];
        ap_uint<192> B_packed[24];
        ap_uint<192> C_packed[packed_size];
        int32_t C_hw[matrix_size];
        int32_t C_sw[matrix_size];
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                A_original[i * BLOCK_SIZE + j] = (i == j) ? 1 : 0;
                B_original[i * BLOCK_SIZE + j] = (i == j) ? 1 : 0;
            }
        }
        
        pack_A(A_original, A_packed);
        pack_B(B_original, B_packed);
        
        mm_pipeline(A_packed, B_packed, C_packed);
        unpack_C(C_packed, C_hw);
        rearrange_C(C_hw, C_hw);
        software_matmul(A_original, B_original, C_sw, BLOCK_SIZE);
        
        if (!compare_matrices(C_hw, C_sw, BLOCK_SIZE)) {
            std::cout << "Identity matrix test FAILED!" << std::endl;
            error_count++;
        } else {
            std::cout << "Identity matrix test PASSED!" << std::endl;
        }
    }
    
    return error_count;
}

int main() {
    std::cout << "Starting mm_pipeline Testbench" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int error_count = 0;
    
    // 运行所有测试
    error_count += test_random_matrices();
    error_count += test_edge_cases();
    
    std::cout << "\n========================================" << std::endl;
    if (error_count == 0) {
        std::cout << "ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << error_count << " test(s) FAILED!" << std::endl;
    }
    
    return error_count;
}
