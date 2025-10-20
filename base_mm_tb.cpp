#include "ap_int.h"
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

void mm_pipeline(
    ap_uint<48>* A,
    ap_uint<48>* B,
    ap_uint<192>* C
);

#define TILE_SIZE 6
#define BLOCK_SIZE 24

void generate_random_matrix(int8_t* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 256 - 128; // -128 到 127
    }
}

void rearrange_and_pack_A(int8_t* A_original, ap_uint<48>* A_packed) {
    int packed_index = 0;
    
    for (int block_i = 0; block_i < BLOCK_SIZE; block_i += TILE_SIZE) {
        for (int block_j = 0; block_j < BLOCK_SIZE; block_j += TILE_SIZE) {
            for (int tile_row = 0; tile_row < TILE_SIZE; tile_row++) {
                ap_uint<48> packed_value = 0;
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

void rearrange_and_pack_B(int8_t* B_original, ap_uint<48>* B_packed) {
    int packed_index = 0;
    
    for (int block_j = 0; block_j < BLOCK_SIZE; block_j += TILE_SIZE) {
        for (int block_i = 0; block_i < BLOCK_SIZE; block_i += TILE_SIZE) {
            for (int tile_col = 0; tile_col < TILE_SIZE; tile_col++) {
                ap_uint<48> packed_value = 0;
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

void unpack_C(ap_uint<192>* C_packed, int32_t* C_unpacked) {
    int unpacked_index = 0;
    
    for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / 6; i++) {
        ap_uint<192> packed_data = C_packed[i];
        for (int j = 0; j < 6; j++) {
            if (unpacked_index < BLOCK_SIZE * BLOCK_SIZE) {
                C_unpacked[unpacked_index++] = (int32_t)packed_data.range(32 * j + 31, 32 * j);
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

void print_matrix_int8(int8_t* matrix, int rows, int cols, const char* name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << (int)matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_matrix_int32(int32_t* matrix, int rows, int cols, const char* name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_packed_A(ap_uint<48>* A_packed, int size, const char* name) {
    std::cout << name << " (packed):" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << "[" << i << "]: ";
        for (int j = 0; j < 6; j++) {
            int8_t value = (int8_t)A_packed[i].range(8 * j + 7, 8 * j);
            std::cout << (int)value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int test_mm_pipeline() {
    std::cout << "\n=== Testing mm_pipeline ===" << std::endl;

    int8_t A_original[BLOCK_SIZE * BLOCK_SIZE];
    int8_t B_original[BLOCK_SIZE * BLOCK_SIZE];
    
    const int packed_A_size = (BLOCK_SIZE * BLOCK_SIZE) / 6;  // 每个 ap_uint<48> 包含 6 个元素
    const int packed_B_size = (BLOCK_SIZE * BLOCK_SIZE) / 6;
    const int packed_C_size = (BLOCK_SIZE * BLOCK_SIZE) / 6;  // 每个 ap_uint<192> 包含 6 个元素
    
    ap_uint<48> A_packed[packed_A_size];
    ap_uint<48> B_packed[packed_B_size];
    ap_uint<192> C_packed[packed_C_size];
    
    int32_t C_hw[BLOCK_SIZE * BLOCK_SIZE];
    int32_t C_sw[BLOCK_SIZE * BLOCK_SIZE];

    srand(42);

    generate_random_matrix(A_original, BLOCK_SIZE * BLOCK_SIZE);
    generate_random_matrix(B_original, BLOCK_SIZE * BLOCK_SIZE);
    print_matrix_int8(A_original, BLOCK_SIZE, BLOCK_SIZE, "Matrix A Original");

    rearrange_and_pack_A(A_original, A_packed);
    rearrange_and_pack_B(B_original, B_packed);
    print_packed_A(A_packed, packed_A_size, "Matrix A Packed");

    std::cout << "Calling mm_pipeline..." << std::endl;
    mm_pipeline(A_packed, B_packed, C_packed);

    unpack_C(C_packed, C_hw);
    print_matrix_int32(C_hw, BLOCK_SIZE, BLOCK_SIZE, "Matrix C from Hardware");

    std::cout << "Running software verification..." << std::endl;
    software_matmul(A_original, B_original, C_sw, BLOCK_SIZE);
    print_matrix_int32(C_sw, BLOCK_SIZE, BLOCK_SIZE, "Matrix C from Software");

    bool success = compare_matrices(C_hw, C_sw, BLOCK_SIZE);

    if (success) {
        std::cout << "mm_pipeline test PASSED!" << std::endl;
    } else {
        std::cout << "mm_pipeline test FAILED!" << std::endl;
        
        std::cout << "First 10 elements comparison:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "C_hw[" << i << "] = " << C_hw[i] 
                      << ", C_sw[" << i << "] = " << C_sw[i] << std::endl;
        }
    }

    return success ? 0 : 1;
}

// 测试边界情况
int test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases for mm_pipeline ===" << std::endl;
    
    int error_count = 0;
    
    // 测试1: 全零矩阵
    std::cout << "Test 1: Zero matrices" << std::endl;
    {
        int8_t A_original[BLOCK_SIZE * BLOCK_SIZE];
        int8_t B_original[BLOCK_SIZE * BLOCK_SIZE];
        ap_uint<48> A_packed[(BLOCK_SIZE * BLOCK_SIZE) / 6];
        ap_uint<48> B_packed[(BLOCK_SIZE * BLOCK_SIZE) / 6];
        ap_uint<192> C_packed[(BLOCK_SIZE * BLOCK_SIZE) / 6];
        int32_t C_hw[BLOCK_SIZE * BLOCK_SIZE];
        int32_t C_sw[BLOCK_SIZE * BLOCK_SIZE];
        
        for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
            A_original[i] = 0;
            B_original[i] = 0;
        }
        
        rearrange_and_pack_A(A_original, A_packed);
        rearrange_and_pack_B(B_original, B_packed);
        
        mm_pipeline(A_packed, B_packed, C_packed);
        unpack_C(C_packed, C_hw);
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
        int8_t A_original[BLOCK_SIZE * BLOCK_SIZE];
        int8_t B_original[BLOCK_SIZE * BLOCK_SIZE];
        ap_uint<48> A_packed[(BLOCK_SIZE * BLOCK_SIZE) / 6];
        ap_uint<48> B_packed[(BLOCK_SIZE * BLOCK_SIZE) / 6];
        ap_uint<192> C_packed[(BLOCK_SIZE * BLOCK_SIZE) / 6];
        int32_t C_hw[BLOCK_SIZE * BLOCK_SIZE];
        int32_t C_sw[BLOCK_SIZE * BLOCK_SIZE];
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                A_original[i * BLOCK_SIZE + j] = (i == j) ? 1 : 0;
                B_original[i * BLOCK_SIZE + j] = (i == j) ? 1 : 0;
            }
        }
        
        rearrange_and_pack_A(A_original, A_packed);
        rearrange_and_pack_B(B_original, B_packed);
        
        mm_pipeline(A_packed, B_packed, C_packed);
        unpack_C(C_packed, C_hw);
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
    error_count += test_mm_pipeline();
    error_count += test_edge_cases();
    
    std::cout << "\n========================================" << std::endl;
    if (error_count == 0) {
        std::cout << "ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << error_count << " test(s) FAILED!" << std::endl;
    }
    
    return error_count;
}
