#include <iostream>
#include <vector>

int main() {
    const int N = 4;  // 4x4矩阵
    std::vector<int> matrix(N * N);
    
    // 初始化矩阵，索引为 i*4+j
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = i * N + j;  // 存储索引值便于验证
        }
    }
    
    std::cout << "矩阵访问顺序：" << std::endl;
    
    // 三层循环构建
    for (int i = 0; i < N; i++) {           // 外层循环：行索引
        for (int k = 0; k < N; k++) {       // 中层循环：每行重复次数
            for (int j = 0; j < N; j++) {   // 内层循环：列索引
                int index = i * N + j;      // 一维数组索引
                std::cout << "(" << i << ", " << j << ") -> 索引[" << index 
                          << "] = " << matrix[index] << std::endl;
            }
        }
    }
    
    return 0;
}
