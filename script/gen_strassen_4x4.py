import numpy as np

def generate_strassen_4x4():

    V2 = np.array([
        [1, 1, 0, 0],    # M1 = (A11+A12)(B11+B22)
        [0, 1, 1, 0],    # M2 = (A21+A22)B11
        [1, 0, 0, 1],    # M3 = A11(B12-B22)
        [0, 0, 1, -1],   # M4 = A22(B21-B11)
        [1, 1, 0, 0],    # M5 = (A11+A12)B22
        [0, -1, -1, 0],  # M6 = (A21-A11)(B11+B12)
        [-1, 0, 0, 1]    # M7 = (A12-A22)(B21+B22)
    ], dtype=int)

    U2 = np.array([
        [1, 0, 0, 1],    # M1 = (A11+A12)(B11+B22)
        [1, 0, 0, 0],    # M2 = (A21+A22)B11
        [0, 1, 0, -1],   # M3 = A11(B12-B22)
        [-1, 1, 0, 0],   # M4 = A22(B21-B11)
        [0, 0, 1, 0],    # M5 = (A11+A12)B22
        [1, 1, 0, 0],    # M6 = (A21-A11)(B11+B12)
        [0, 0, 1, 1]     # M7 = (A12-A22)(B21+B22)
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, 0, 0, 1],    # C11 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],    # C12 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],    # C21 = M2 + M4
        [1, -1, 0, 0, 0, 1, 0]    # C22 = M1 - M2 + M6
    ], dtype=int)

    V4 = np.kron(V2, V2)  # 49×16：7×7块，每块4×4
    U4 = np.kron(U2, U2)  # 49×16：7×7块，每块4×4
    W4 = np.kron(W2, W2)  # 16×49：4×4块，每块7×7

    return V2, U2, W2, V4, U4, W4

if __name__ == "__main__":
    V2, U2, W2, V4, U4, W4 = generate_strassen_4x4()

    print("=== 矩阵维度 ===")
    print(f"V2: {V2.shape}, U2: {U2.shape}, W2: {W2.shape}")
    print(f"V4: {V4.shape}, U4: {U4.shape}, W4: {W4.shape}\n")

    print("=== 2×2基础系数矩阵 V2 ===")
    print(V2, "\n")

    print("=== 4×4系数矩阵 V4 前7行前8列（示例） ===")
    print(V4[:7, :8], "\n")  # V4的前7行对应V2第一行的Kronecker积块

    print("=== 4×4系数矩阵 W4 前4行前14列（示例） ===")
    print(W4[:4, :14], "\n")  # W4的前4行对应W2第一行的Kronecker积块

    np.savetxt("V4_matrix.txt", V4, fmt="%d")
    np.savetxt("U4_matrix.txt", U4, fmt="%d")
    np.savetxt("W4_matrix.txt", W4, fmt="%d")
    print("完整矩阵已保存到文本文件（V4_matrix.txt, U4_matrix.txt, W4_matrix.txt）")