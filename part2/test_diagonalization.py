"""
    Bộ kiểm thử (test suite) cho module diagonalization.py.
    Tập trung kiểm thử các hàm:
        1. qr_factorization_gram_schmidt()
        2. eigen_qr_iteration()
        3. diagonalize_matrix()

    Phân loại test cases:
        - Tests 1-2 : Kiểm tra phân rã QR (Unit Test cho Gram-Schmidt)
        - Tests 3-5 : Kiểm tra tính đúng đắn chéo hóa (Correctness)
        - Tests 6-7 : Kiểm tra các trường hợp biên (Edge Cases)
"""

import numpy as np
# Import các hàm cần thiết từ file của bạn
from diagonalization import qr_factorization_gram_schmidt, eigen_qr_iteration, diagonalize_matrix
#-----------------------------------
# HÀM TIỆN ÍCH (Utility Functions)
#-------------------------------------

def is_orthogonal(P, tol=1e-9):
    """Kiểm tra P có phải ma trận trực giao (P * P^T = I) không."""
    n = P.shape[1]
    I = np.eye(n)
    return np.allclose(P.T @ P, I, atol=tol)

def is_upper_triangular(R, tol=1e-9):
    """Kiểm tra R có phải ma trận tam giác trên không."""
    return np.allclose(R, np.triu(R), atol=tol)

# ---------------------------
# UNIT TESTS - KIỂM TRA PHÂN RÃ QR
# -------------------------

def test_1_qr_gram_schmidt():
    """
    Kiểm tra hàm qr_factorization_gram_schmidt.
    Yêu cầu: A = Q * R, Q trực giao, R tam giác trên.
    """
    A = np.array([[1, 2], [3, 4]], dtype=float)
    Q, R = qr_factorization_gram_schmidt(A)
    
    assert np.allclose(Q @ R, A, atol=1e-9), "QR: Tái tạo A thất bại!"
    assert is_orthogonal(Q), "QR: Q không trực giao!"
    assert is_upper_triangular(R), "QR: R không phải tam giác trên!"
    print("TEST 1 PASSED — Unit Test QR Gram-Schmidt")

# -------------------------------
# CORE TESTS - KIỂM TRA CHÉO HÓA
# ------------------------------

def test_2_diagonalize_symmetric():
    """
    Kiểm tra diagonalize_matrix với ma trận đối xứng 3x3.
    """
    A = np.array([[4, 2, 2], [2, 4, 2], [2, 2, 4]], dtype=float)
    P, D, P_inv = diagonalize_matrix(A)
    
    # Kiểm tra định nghĩa chéo hóa: A = P * D * P_inv
    assert np.allclose(P @ D @ P_inv, A, atol=1e-7), "Chéo hóa: P*D*P_inv != A"
    # Với ma trận đối xứng, P từ QR iteration phải trực giao
    assert is_orthogonal(P), "Chéo hóa: P không trực giao"
    print("TEST 2 PASSED — Chéo hóa ma trận đối xứng")

def test_3_eigenvalues_accuracy():
    """
    So sánh trị riêng từ eigen_qr_iteration với NumPy.
    """
    A = np.array([[6, 2], [2, 3]], dtype=float)
    lambdas, V = eigen_qr_iteration(A, iterations=1000)
    
    expected_lambdas = np.sort(np.linalg.eigvals(A))
    calculated_lambdas = np.sort(lambdas)
    
    assert np.allclose(calculated_lambdas, expected_lambdas, atol=1e-5), \
        f"Trị riêng sai! Kỳ vọng {expected_lambdas}, tính ra {calculated_lambdas}"
    print("TEST 3 PASSED — Độ chính xác trị riêng (vs NumPy)")

def test_4_identity():
    """Kiểm tra với ma trận đơn vị."""
    A = np.eye(4)
    P, D, P_inv = diagonalize_matrix(A)
    assert np.allclose(D, A), "Ma trận đơn vị phải giữ nguyên D"
    print("TEST 4 PASSED — Ma trận đơn vị")

# -----------------------
# EDGE CASE TESTS
# ------------------------

def test_5_singular_matrix():
    """Ma trận suy biến (có trị riêng bằng 0)."""
    A = np.array([[1, 2], [2, 4]], dtype=float) # det = 0
    P, D, P_inv = diagonalize_matrix(A)
    
    evals = np.diag(D)
    assert any(np.isclose(evals, 0, atol=1e-7)), "Thiếu trị riêng 0 cho ma trận suy biến"
    assert np.allclose(P @ D @ P_inv, A, atol=1e-7), "Tái tạo thất bại trên ma trận suy biến"
    print("TEST 5 PASSED — Ma trận suy biến")

def test_6_non_square():
    """Kiểm tra bắt lỗi ma trận không vuông."""
    A = np.zeros((2, 3))
    result = diagonalize_matrix(A)
    assert "Lỗi" in result, "Chưa xử lý trường hợp ma trận không vuông"
    print("TEST 6 PASSED — Bắt lỗi ma trận không vuông")

# --------------------------
# EXECUTION
# ---------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING TESTS FOR: diagonalization.py")
    print("=" * 60)

    test_list = [
        test_1_qr_gram_schmidt,
        test_2_diagonalize_symmetric,
        test_3_eigenvalues_accuracy,
        test_4_identity,
        test_5_singular_matrix,
        test_6_non_square
    ]

    passed = 0
    for test in test_list:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"{test.__name__} FAILED: {e}")
        except Exception as e:
            print(f"{test.__name__} ERROR: {type(e).__name__}: {e}")

    print(f"\nSummary: {passed}/{len(test_list)} tests passed.")
    print("=" * 60)