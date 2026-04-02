"""
Module: decomposition.py
Description:
    Triển khai thuật toán Phân rã Cholesky (Cholesky Decomposition).
    Cho ma trận A đối xứng xác định dương (Symmetric Positive Definite - SPD),
    thuật toán phân rã A thành tích L * L^T, trong đó L là ma trận tam giác dưới.

    Độ phức tạp: O(n^3 / 3)

"""

import math
import numpy as np


def cholesky_decomposition(A):
    """
    Phân rã Cholesky: A = L * L^T.

    Tham số (Parameters):
        A (list[list[float]]): Ma trận vuông n×n, đối xứng xác định dương (SPD).

    Trả về (Returns):
        list[list[float]]: Ma trận tam giác dưới L sao cho A = L * L^T.

    Ngoại lệ (Raises):
        ValueError: Nếu A không xác định dương (giá trị trong căn bậc hai <= 0).

    Ví dụ (Example):
        >>> A = [[4, 2], [2, 10]]
        >>> L = cholesky_decomposition(A)
        >>> # L = [[2.0, 0.0], [1.0, 3.0]]
    """
    n = len(A)

    # EDGE CASE: Kiểm tra tính đối xứng của ma trận
    for i in range(n):
        for j in range(n):
            # Dùng math.isclose để so sánh số thực, tránh sai số máy tính
            if not math.isclose(A[i][j], A[j][i], rel_tol=1e-9):
                raise ValueError(
                    "Lỗi: Ma trận không đối xứng (Not Symmetric)."
                )

    # Khởi tạo ma trận L (n×n) toàn số 0
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            # Tính tổng tích lũy: sum_k = Σ(k=0..j-1) L[·][k] * L[·][k]
            sum_k = 0.0

            if j == i:
                # --- Phần tử TRÊN đường chéo chính: L[j][j] ---
                # Công thức: L[j][j] = sqrt( A[j][j] - Σ(k=0..j-1) L[j][k]^2 )
                for k in range(j):
                    sum_k += L[j][k] ** 2

                # EDGE CASE: Kiểm tra điều kiện SPD (biểu thức trong căn phải > 0)
                if A[j][j] - sum_k <= 0:
                    raise ValueError(
                        "Lỗi: Ma trận không xác định dương (Not Positive Definite)."
                    )

                L[j][j] = math.sqrt(A[j][j] - sum_k)

            else:
                # --- Phần tử DƯỚI đường chéo chính: L[i][j] (i > j) ---
                # Công thức: L[i][j] = (A[i][j] - Σ(k=0..j-1) L[i][k]*L[j][k]) / L[j][j]
                for k in range(j):
                    sum_k += L[i][k] * L[j][k]

                L[i][j] = (A[i][j] - sum_k) / L[j][j]

    return L


# =============================================================================
# PHẦN KIỂM CHỨNG (Chỉ chạy khi thực thi trực tiếp file này)
# =============================================================================
if __name__ == "__main__":
    print("Đang kiểm tra thuật toán Phân rã Cholesky...\n")

    # 1. Tạo ma trận SPD 4×4 từ ma trận ngẫu nhiên: A = X * X^T
    #    (Tích X*X^T luôn cho kết quả là ma trận SPD)
    np.random.seed(42)  # Cố định seed để kết quả tái lập được (reproducible)
    X = np.random.rand(4, 4)
    A_spd = np.dot(X, X.T)

    # 2. Phân rã bằng hàm tự triển khai
    L_custom = cholesky_decomposition(A_spd.tolist())

    # 3. Phân rã bằng hàm chuẩn của NumPy (đối chiếu)
    L_numpy = np.linalg.cholesky(A_spd)

    # 4. In kết quả hai phương pháp
    print("Ma trận L (Tự code):")
    for row in L_custom:
        print([f"{val:.4f}" for val in row])

    print("\nMa trận L (NumPy):")
    for row in L_numpy.tolist():
        print([f"{val:.4f}" for val in row])

    # 5. So sánh kết quả (tolerance mặc định: atol=1e-8, rtol=1e-5)
    is_correct = np.allclose(L_custom, L_numpy)
    print(f"\n=> Kết quả so sánh (True là code chuẩn): {is_correct}")
