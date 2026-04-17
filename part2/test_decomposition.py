"""
Module: test_decomposition.py
Description:
    Bộ kiểm thử (test suite) cho hàm cholesky_decomposition()
    trong module decomposition.py.

    Phân loại test cases:
        - Tests 1-5 : Kiểm tra tính đúng đắn (Correctness Tests)
        - Tests 6-8 : Kiểm tra các trường hợp biên (Edge Case Tests)

    Phương pháp kiểm chứng:
        1. So sánh L với kết quả biết trước (hand-calculated)
        2. Kiểm tra L có phải tam giác dưới (lower triangular)
        3. Kiểm tra tái tạo: L * L^T == A (reconstruction)
        4. Đối chiếu với NumPy (np.linalg.cholesky)

    Cách chạy:
        python part2/test_decomposition.py
"""

import math
import numpy as np
from decomposition import cholesky_decomposition


# =============================================================================
# HÀM TIỆN ÍCH (Utility Functions)
# =============================================================================

def multiply_L_Lt(L):
    """
    Tính tích L * L^T (reconstruction) từ ma trận tam giác dưới L.

    Tham số (Parameters):
        L (list[list[float]]): Ma trận tam giác dưới n×n.

    Trả về (Returns):
        list[list[float]]: Ma trận n×n kết quả của phép nhân L * L^T.

    Công thức:
        result[i][j] = Σ(k=0..n-1) L[i][k] * L[j][k]
    """
    n = len(L)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += L[i][k] * L[j][k]  # (L * L^T)[i][j]
    return result


def matrices_are_close(A, B, tol=1e-9):
    """
    So sánh 2 ma trận (list of lists) theo từng phần tử với sai số cho phép.

    Tham số (Parameters):
        A (list[list[float]]): Ma trận thứ nhất.
        B (list[list[float]]): Ma trận thứ hai (cùng kích thước với A).
        tol (float): Sai số tuyệt đối tối đa cho phép. Mặc định: 1e-9.

    Trả về (Returns):
        bool: True nếu |A[i][j] - B[i][j]| <= tol với mọi i, j.
    """
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


def is_lower_triangular(L, tol=1e-12):
    """
    Kiểm tra ma trận L có phải là tam giác dưới (lower triangular) không.

    Tham số (Parameters):
        L (list[list[float]]): Ma trận vuông n×n cần kiểm tra.
        tol (float): Ngưỡng coi là bằng 0. Mặc định: 1e-12.

    Trả về (Returns):
        bool: True nếu mọi phần tử phía trên đường chéo chính đều ≈ 0.
              Tức là L[i][j] ≈ 0 với mọi j > i.
    """
    n = len(L)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(L[i][j]) > tol:
                return False
    return True


# =============================================================================
# CÁC TEST CASES — KIỂM TRA TÍNH ĐÚNG ĐẮN (Correctness Tests)
# =============================================================================

def test_1_matrix_1x1():
    """
    TEST 1: Ma trận 1×1 — trường hợp nhỏ nhất (base case).

    Input:  A = [[9]]
    Expect: L = [[3]]  (vì sqrt(9) = 3, và 3 × 3 = 9)
    Verify: Kết quả chính xác + L là tam giác dưới.
    """
    A = [[9.0]]
    L = cholesky_decomposition(A)

    assert L == [[3.0]], f"Sai! L = {L}, kỳ vọng [[3.0]]"
    assert is_lower_triangular(L), "L không phải tam giác dưới!"
    print("TEST 1 PASSED — Ma trận 1×1")


def test_2_matrix_2x2():
    """
    TEST 2: Ma trận 2×2 với kết quả tính tay (hand-calculated).

    Input:  A = [[4, 2],      Expect: L = [[2, 0],
                 [2, 10]]                   [1, 3]]

    Verify: (1) L khớp kết quả kỳ vọng
            (2) L là tam giác dưới
            (3) L * L^T tái tạo lại đúng A (reconstruction check)
    """
    A = [[4.0, 2.0],
         [2.0, 10.0]]

    L = cholesky_decomposition(A)

    L_expected = [[2.0, 0.0],
                  [1.0, 3.0]]

    # Kiểm tra kết quả khớp giá trị kỳ vọng
    assert matrices_are_close(L, L_expected), \
        f"Sai! L = {L}, kỳ vọng {L_expected}"
    assert is_lower_triangular(L), "L không phải tam giác dưới!"

    # Kiểm tra tái tạo: L * L^T phải bằng A
    A_reconstructed = multiply_L_Lt(L)
    assert matrices_are_close(A, A_reconstructed), \
        "L * L^T ≠ A!"
    print("TEST 2 PASSED — Ma trận 2×2 (kết quả biết trước)")


def test_3_identity_matrix_3x3():
    """
    TEST 3: Ma trận đơn vị I₃ (identity matrix).

    Input:  A = I₃
    Expect: L = I₃  (vì I = I × I^T, phân rã tầm thường)
    Verify: Kết quả chính xác + L là tam giác dưới.
    """
    A = [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]

    L = cholesky_decomposition(A)

    L_expected = [[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]

    assert matrices_are_close(L, L_expected), \
        f"Sai! L = {L}, kỳ vọng {L_expected}"
    assert is_lower_triangular(L), "L không phải tam giác dưới!"
    print("TEST 3 PASSED — Ma trận đơn vị I₃")


def test_4_matrix_3x3_general():
    """
    TEST 4: Ma trận SPD 3×3 tổng quát (có phần tử âm).

    Input:  A = [[25, 15, -5],    Expect: L = [[ 5, 0, 0],
                 [15, 18,  0],                  [ 3, 3, 0],
                 [-5,  0, 11]]                  [-1, 1, 3]]

    Verify: (1) L khớp kết quả kỳ vọng
            (2) L là tam giác dưới
            (3) Reconstruction: L * L^T == A
    """
    A = [[25.0, 15.0, -5.0],
         [15.0, 18.0,  0.0],
         [-5.0,  0.0, 11.0]]

    L = cholesky_decomposition(A)

    L_expected = [[5.0, 0.0, 0.0],
                  [3.0, 3.0, 0.0],
                  [-1.0, 1.0, 3.0]]

    assert matrices_are_close(L, L_expected), \
        f"Sai! L = {L}, kỳ vọng {L_expected}"
    assert is_lower_triangular(L), "L không phải tam giác dưới!"

    # Kiểm tra tái tạo: L * L^T phải bằng A
    A_reconstructed = multiply_L_Lt(L)
    assert matrices_are_close(A, A_reconstructed), "L * L^T ≠ A!"
    print("TEST 4 PASSED — Ma trận SPD 3×3 tổng quát")


def test_5_random_5x5_vs_numpy():
    """
    TEST 5: Ma trận ngẫu nhiên 5×5 — đối chiếu với thư viện chuẩn NumPy.

    Phương pháp:
        1. Tạo A = X × X^T (tích này luôn cho ma trận SPD)
        2. Phân rã bằng hàm tự code vs np.linalg.cholesky
        3. So sánh hai kết quả với tolerance = 1e-10

    Seed: np.random.seed(123) để kết quả tái lập được (reproducible).
    """
    np.random.seed(123)  # Cố định seed cho tính tái lập
    X = np.random.rand(5, 5)
    A = np.dot(X, X.T)  # A = X * X^T → luôn SPD

    # So sánh kết quả tự code vs NumPy
    L_custom = cholesky_decomposition(A.tolist())
    L_numpy = np.linalg.cholesky(A)

    assert np.allclose(L_custom, L_numpy, atol=1e-10), \
        "Kết quả khác NumPy!"
    assert is_lower_triangular(L_custom), "L không phải tam giác dưới!"
    print("TEST 5 PASSED — Ma trận ngẫu nhiên 5×5 (đối chiếu NumPy)")


# =============================================================================
# CÁC TEST CASES — KIỂM TRA TRƯỜNG HỢP BIÊN (Edge Case Tests)
# =============================================================================

def test_6_not_positive_definite():
    """
    TEST 6 (Edge Case): Ma trận đối xứng nhưng KHÔNG xác định dương.

    Input:  A = [[1, 2],
                 [2, 1]]
    Eigenvalues: λ₁ = -1, λ₂ = 3 → Có eigenvalue âm → Không SPD.
    Expect: Hàm phải raise ValueError.
    """
    A = [[1.0, 2.0],
         [2.0, 1.0]]

    try:
        cholesky_decomposition(A)
        # Nếu chạy tới đây mà không raise → test FAIL
        assert False, "Không raise ValueError cho ma trận không xác định dương!"
    except ValueError:
        pass  # Đúng kỳ vọng: hàm đã phát hiện và raise lỗi
    print("TEST 6 PASSED — Phát hiện ma trận không xác định dương")


def test_7_diagonal_matrix():
    """
    TEST 7: Ma trận đường chéo (diagonal matrix) — trường hợp đặc biệt.

    Input:  A = diag(4, 9, 16)
    Expect: L = diag(2, 3, 4)   (vì sqrt(4)=2, sqrt(9)=3, sqrt(16)=4)

    Lý do test: Ma trận đường chéo là SPD đơn giản nhất,
                L cũng phải là đường chéo (tất cả phần tử ngoài đường chéo = 0).
    """
    A = [[4.0, 0.0, 0.0],
         [0.0, 9.0, 0.0],
         [0.0, 0.0, 16.0]]

    L = cholesky_decomposition(A)

    L_expected = [[2.0, 0.0, 0.0],
                  [0.0, 3.0, 0.0],
                  [0.0, 0.0, 4.0]]

    assert matrices_are_close(L, L_expected), \
        f"Sai! L = {L}, kỳ vọng {L_expected}"
    assert is_lower_triangular(L), "L không phải tam giác dưới!"
    print("TEST 7 PASSED — Ma trận đường chéo")


def test_8_not_symmetric():
    """
    TEST 8 (Edge Case): Ma trận KHÔNG đối xứng (not symmetric).

    Input:  A = [[1, 2],
                 [3, 4]]
    Vấn đề: A[0][1] = 2 ≠ A[1][0] = 3 → Ma trận không đối xứng.
    Expect: Hàm phải raise ValueError (vi phạm điều kiện đầu vào SPD).
    """
    A = [[1.0, 2.0],
         [3.0, 4.0]]

    try:
        cholesky_decomposition(A)
        # Nếu chạy tới đây mà không raise → test FAIL
        assert False, "Không raise ValueError cho ma trận không đối xứng!"
    except ValueError:
        pass  # Đúng kỳ vọng: hàm đã phát hiện và raise lỗi
    print("TEST 8 PASSED — Phát hiện ma trận không đối xứng")

def test_9_not_square_matrix():
    """
    TEST 9 (Edge Case): Ma trận KHÔNG vuông (chữ nhật).

    Input:  A = [[1, 2, 3],
                 [4, 5, 6]]
    Vấn đề: Số hàng (2) khác số cột (3).
    Expect: Hàm phải raise ValueError do không phải ma trận vuông.
    """
    A = [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

    try:
        cholesky_decomposition(A)
        # Nếu chạy tới đây mà không raise -> test FAIL
        assert False, "Không raise ValueError cho ma trận chữ nhật!"
    except ValueError as e:
        # Tùy chọn: có thể in ra thông báo lỗi để kiểm tra câu chữ
        # print(f"  [Đúng kỳ vọng] Bắt được lỗi: {e}")
        pass
    print("TEST 9 PASSED — Phát hiện ma trận không vuông")


# =============================================================================
# Chạy tất cả test cases và tổng hợp kết quả
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("KIỂM THỬ — Phân rã Cholesky (cholesky_decomposition)")
    print("=" * 55 + "\n")

    # Danh sách tất cả test cases (thứ tự chạy)
    tests = [
        test_1_matrix_1x1,
        test_2_matrix_2x2,
        test_3_identity_matrix_3x3,
        test_4_matrix_3x3_general,
        test_5_random_5x5_vs_numpy,
        test_6_not_positive_definite,
        test_7_diagonal_matrix,
        test_8_not_symmetric,
        test_9_not_square_matrix,
    ]

    # Bộ đếm kết quả
    passed = 0
    failed = 0

    # Chạy từng test, bắt exception để không dừng toàn bộ suite
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            # Test assertion thất bại → logic sai
            print(f"{test_fn.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            # Lỗi không mong đợi (runtime error, v.v.)
            print(f"{test_fn.__name__} ERROR: {type(e).__name__}: {e}")
            failed += 1

    # Tổng hợp kết quả
    print(f"\n{'=' * 55}")
    print(f"KẾT QUẢ: {passed}/{passed + failed} tests passed")
