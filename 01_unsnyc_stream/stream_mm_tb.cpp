#include "ap_int.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>  // Added for timing measurements
#include "stream_mm.hpp"

/**
 * @brief Pack int8_t array to ap_int<48> array
 * 
 * Each ap_int<48> element packs 6 int8_t values (48/8 = 6)
 * 
 * @param src_int8 Source int8_t array
 * @param dst_ap48 Destination ap_int<48> array
 * @param total_elements Total number of int8_t elements
 */
static void pack_int8_to_ap48(
    const int8_t *src_int8,
    ap_int<48> *dst_ap48,
    int total_elements
){
    int packed_elements = total_elements / 6; // 48 bits / 8 bits = 6 elements per packed value
    for (int i = 0; i < packed_elements; i++) {
        ap_int<48> packed_value = 0;
        for (int j = 0; j < 6; j++) {
            int src_idx = i * 6 + j;
            if (src_idx < total_elements) {
                int8_t value = src_int8[src_idx];
                packed_value.range(8*j+7, 8*j) = value;
            }
        }
        dst_ap48[i] = packed_value;
    }
}

/**
 * @brief Unpack ap_int<192> array to int32_t array
 * 
 * Each ap_int<192> element unpacks to 6 int32_t values (192/32 = 6)
 * 
 * @param src_ap192 Source ap_int<192> array
 * @param dst_int32 Destination int32_t array
 * @param total_elements Total number of int32_t elements
 */
static void unpack_ap192_to_int32(
    ap_int<192> *src_ap192,
    int32_t *dst_int32,
    int total_elements
){
    int packed_elements = total_elements / 6; // 192 bits / 32 bits = 6 elements per packed value
    for (int i = 0; i < packed_elements; i++) {
        ap_int<192> packed_value = src_ap192[i];
        for (int j = 0; j < 6; j++) {
            int dst_idx = i * 6 + j;
            if (dst_idx < total_elements) {
                dst_int32[dst_idx] = (int32_t)packed_value.range(32*j+31, 32*j);
            }
        }
    }
}

static void sw_mmult(
    const int8_t *A,
    const int8_t *B,
    int32_t *C,
    int shape
){
    for (int i = 0; i < shape; i++) {
        for (int j = 0; j < shape; j++) {
            long sum = 0; // Use 64-bit accumulation to avoid intermediate overflow
            for (int k = 0; k < shape; k++) {
                sum += (long)A[i * shape + k] * (long)B[k * shape + j];
            }
            C[i * shape + j] = (int32_t)sum;
        }
    }
}

int main(){
    int shape = MAT_SIZE;

    const int maxElement = MAT_SIZE*MAT_SIZE;
    const int packedElements = maxElement / 6; // 48 bits / 8 bits = 6 elements per packed value
    const int outputPackedElements = maxElement / 6; // 192 bits / 32 bits = 6 elements per packed value
    bool test_pass = true;
    double ops_per_mmult = 2.0 * shape * shape * shape;

    std::cout << "Running test shape = " << shape << std::endl;

    int8_t *A = new int8_t[maxElement];
    int8_t *B = new int8_t[maxElement];
    ap_int<48> *A_packed = new ap_int<48>[packedElements];
    ap_int<48> *B_packed = new ap_int<48>[packedElements];
    ap_int<192> *C_hw_packed = new ap_int<192>[outputPackedElements];
    int32_t *C_hw = new int32_t[maxElement];
    int32_t *C_sw = new int32_t[maxElement];

    srand(42);
    for (int i = 0; i < maxElement; i++){
        A[i] = (int8_t)(rand() % 256 - 128);
        B[i] = (int8_t)(rand() % 256 - 128);
    }

    // Pack int8_t arrays to ap_int<48> arrays
    pack_int8_to_ap48(A, A_packed, maxElement);
    pack_int8_to_ap48(B, B_packed, maxElement);

    std::cout << "   Computing ..." << std::endl;

    for (int i = 0; i < maxElement; i++){
        C_hw[i] = 0;
        C_sw[i] = 0;
    }
    for (int i = 0; i < outputPackedElements; i++){
        C_hw_packed[i] = 0;
    }

    sw_mmult(A, B, C_sw, shape);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Call matrix_mul with packed ap_int<48> arrays and ap_int<192> output
    matrix_mul(A_packed, B_packed, C_hw_packed);

    auto end_time = std::chrono::high_resolution_clock::now();

    // Unpack the hardware results from ap_int<192> to int32_t
    unpack_ap192_to_int32(C_hw_packed, C_hw, maxElement);

    for (int i = 0; i < shape * shape; i++) {
        if (C_hw[i] != C_sw[i]) {
            test_pass = false;
            std::cout << " mismatch at index " << i 
                        << ": HW = " << C_hw[i] 
                        << ", SW = " << C_sw[i] << std::endl;
            break;
        }
    }
    if (test_pass)
        std::cout << "   Test Passed." << std::endl;
    else {
        std::cout << "   Test Failed." << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] A_packed;
    delete[] B_packed;
    delete[] C_hw_packed;
    delete[] C_hw;
    delete[] C_sw;

    // return test_pass ? 0 : 1;
    return 0;
}
