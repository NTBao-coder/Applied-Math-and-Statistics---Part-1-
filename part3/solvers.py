"""
solvers.py — Các phương pháp giải hệ Ax = b
Gồm:
  1. Gauss elimination (trực tiếp)
  2. Cholesky solver (trực tiếp, dùng Cholesky từ Part 2)
  3. Gauss-Seidel (phương pháp lặp)

Mỗi phương pháp có implement và kiểm chứng bằng numpy để tối ưu hóa thời gian chạy. 
Giải thích ở testcase n = 1000.
Với Phương pháp Khử Gauss (Gauss Elimination): 
- Mặc dù thuật toán có độ phức tạp là $\mathcal{O}(n^3)$, nhưng trong solve_gauss, 
dòng code quan trọng nhất đã được vector hóa (vectorization) bằng numpy.
- Với Phương pháp Cholesky (Trực tiếp): Thuật toán phân rã cholesky_decomposition ở Part 2 được viết bằng thuần Python thông qua các vòng lặp lồng nhau hoàn toàn (không vector hóa qua NumPy). 
Do đó, phương pháp này sẽ tốn nhiều thời gian nhất (~20 giây) vì Python phải thực hiện từng phép 
cộng trừ nhân chia một cách tuyến tính, tuy nhiên vẫn đảm bảo sai số của Cholesky thu được.
"""

import numpy as np
import sys
import os

# Import từ Part 2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part2'))
from decomposition import cholesky_decomposition


# ===========================================================================
# 1. Gauss Elimination Solver
# ===========================================================================

def solve_gauss(A, b):
    """
    Giải Ax = b bằng phép khử Gauss với partial pivoting.
    
    Input:  A ∈ R^{n×n}, b ∈ R^n
    Output: x ∈ R^n nghiệm của hệ
    
    Thuật toán:
        1. Tạo ma trận tăng cường [A | b]
        2. Khử Gauss với partial pivoting → dạng tam giác trên
        3. Thế ngược (back substitution)
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Gauss solver yêu cầu ma trận vuông.")
    if len(b) != n:
        raise ValueError("Kích thước b không khớp với A.")

    # Ma trận tăng cường [A | b]
    M = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination với partial pivoting
    for k in range(n):
        # Tìm pivot lớn nhất
        pivot_row = k + np.argmax(np.abs(M[k:, k]))
        if abs(M[pivot_row, k]) < 1e-14:
            raise ValueError(f"Ma trận suy biến: pivot ≈ 0 tại cột {k}.")

        # Hoán đổi dòng
        if pivot_row != k:
            M[[k, pivot_row]] = M[[pivot_row, k]]

        # Khử
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]

    return x





# ===========================================================================
# 2.Cholesky Solver
# ===========================================================================

def solve_cholesky(A, b):
    """
    Giải Ax = b bằng phân rã Cholesky.
    
    Input:  A ∈ R^{n×n} (SPD), b ∈ R^n
    Output: x ∈ R^n
    
    Thuật toán:
        1. Phân rã A = L * L^T
        2. Giải Ly = b (forward substitution)
        3. Giải L^T x = y (back substitution)
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    n = A.shape[0]

    # Phân rã Cholesky
    L_list = cholesky_decomposition(A.tolist())
    L = np.array(L_list)
    LT = L.T

    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # Back substitution: L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i+1:n], x[i+1:n])) / LT[i, i]

    return x


# ===========================================================================
# 3. Gauss-Seidel Iterative Solver
# ===========================================================================

def is_diagonally_dominant(A):
    """
    Kiểm tra ma trận có chéo trội chặt hàng không.
    |a_ii| > Σ_{j≠i} |a_ij| với mọi i.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag <= off_diag:
            return False
    return True


def solve_gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=10000):
    """
    Giải Ax = b bằng phương pháp lặp Gauss-Seidel.
    
    Input:  A ∈ R^{n×n}, b ∈ R^n
            x0: nghiệm ban đầu (mặc định = 0)
            tol: sai số dừng
            max_iter: số lần lặp tối đa
    Output: x ∈ R^n, iterations (số lần lặp), history (lịch sử hội tụ)
    
    Công thức lặp (Gauss-Seidel):
        x_i^{(k+1)} = (1/a_ii) * (b_i - Σ_{j<i} a_ij x_j^{(k+1)} - Σ_{j>i} a_ij x_j^{(k)})
    
    Điều kiện hội tụ: A chéo trội chặt hàng hoặc SPD.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    n = A.shape[0]

    # Kiểm tra điều kiện hội tụ
    dd = is_diagonally_dominant(A)
    if not dd:
        # Kiểm tra SPD
        try:
            eigenvals = np.linalg.eigvalsh(A)
            is_spd = np.all(eigenvals > 0) and np.allclose(A, A.T)
        except Exception:
            is_spd = False

        if not is_spd:
            print("⚠ Cảnh báo: Ma trận không chéo trội chặt và không SPD.")
            print("  Gauss-Seidel có thể không hội tụ!")

    # Khởi tạo
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float).flatten()

    history = []  # Lưu sai số tại mỗi lần lặp

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            # Σ_{j<i} a_ij * x_j^{(k+1)}
            s1 = np.dot(A[i, :i], x[:i])
            # Σ_{j>i} a_ij * x_j^{(k)}
            s2 = np.dot(A[i, i+1:], x[i+1:])

            if abs(A[i, i]) < 1e-14:
                raise ValueError(f"Phần tử đường chéo a[{i},{i}] ≈ 0, không thể lặp.")

            x[i] = (b[i] - s1 - s2) / A[i, i]

        # Sai số: ||x^{(k+1)} - x^{(k)}|| / ||x^{(k+1)}||
        diff = np.linalg.norm(x - x_old)
        rel_err = diff / (np.linalg.norm(x) + 1e-16)
        history.append(rel_err)

        if rel_err < tol:
            return x, k + 1, history

    print(f"⚠ Gauss-Seidel chưa hội tụ sau {max_iter} lần lặp (sai số = {rel_err:.2e}).")
    return x, max_iter, history


# ===========================================================================
# 4. Condition Number
# ===========================================================================

def condition_number(A):
    """
    Tính số điều kiện κ₂(A) = σ_max / σ_min.
    
    Input:  A ∈ R^{n×n}
    Output: κ₂(A)
    """
    A = np.array(A, dtype=float)
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[-1] < 1e-15:
        return float('inf')
    return sv[0] / sv[-1]


# ===========================================================================
# 5. Verification & Test Suite
# ===========================================================================

def verify_solver(solver_func, A, b, name="Solver"):
    """So sánh kết quả solver với NumPy."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    result = solver_func(A, b)
    # Handle Gauss-Seidel trả về tuple
    if isinstance(result, tuple):
        x = result[0]
        iters = result[1]
        print(f"  Số lần lặp: {iters}")
    else:
        x = result

    x_np = np.linalg.solve(A, b)

    err = np.linalg.norm(x - x_np) / (np.linalg.norm(x_np) + 1e-16)
    residual = np.linalg.norm(A @ x - b) / (np.linalg.norm(b) + 1e-16)

    print(f"=== {name} ===")
    print(f"  x       = {x}")
    print(f"  x_numpy = {x_np}")
    print(f"  ‖x − x_np‖/‖x_np‖  = {err:.2e}")
    print(f"  ‖Ax − b‖/‖b‖       = {residual:.2e}")
    print(f"  Kết quả: {'PASS ✓' if err < 1e-6 else 'FAIL ✗'}")
    return x


def run_all_tests():
    """Chạy tất cả test cases."""
    print("=" * 60)
    print("  TEST SUITE — GIẢI HỆ PHƯƠNG TRÌNH")
    print("=" * 60)

    np.random.seed(42)

    # --- Gauss Solver ---
    print("\n--- Gauss Elimination ---")
    # Test 1: Hệ 3x3 đơn giản
    A1 = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b1 = np.array([8, -11, -3], dtype=float)
    verify_solver(solve_gauss, A1, b1, "Gauss 3×3")

    # Test 2: Hệ 4x4
    A2 = np.array([[4, 3, 0, 0], [3, 4, 1, 0], [0, 1, 4, 3], [0, 0, 3, 4]], dtype=float)
    b2 = np.array([1, 2, 3, 4], dtype=float)
    verify_solver(solve_gauss, A2, b2, "Gauss 4×4")

    # Test 3: Ma trận đơn vị
    verify_solver(solve_gauss, np.eye(3), np.array([1, 2, 3.0]), "Gauss Identity")

    # Test 4: Ma trận ngẫu nhiên 5x5
    A4 = np.random.randn(5, 5) + 5 * np.eye(5)
    b4 = np.random.randn(5)
    verify_solver(solve_gauss, A4, b4, "Gauss Random 5×5")

    # Test 5: Ma trận cần pivoting
    A5 = np.array([[0, 1, 2], [1, 0, 3], [3, 2, 1]], dtype=float)
    b5 = np.array([1, 2, 3], dtype=float)
    verify_solver(solve_gauss, A5, b5, "Gauss Pivoting")



    # --- Cholesky Solver ---
    print("\n--- Cholesky Solver ---")
    # Tạo ma trận SPD cho test
    M_chol = np.random.randn(4, 4)
    A_chol = M_chol @ M_chol.T + np.eye(4)
    b_chol = np.random.randn(4)
    verify_solver(solve_cholesky, A_chol, b_chol, "Cholesky SPD 4×4")
    verify_solver(solve_cholesky, np.eye(3), np.array([1, 2, 3.0]), "Cholesky Identity")

    # --- Gauss-Seidel ---
    print("\n--- Gauss-Seidel ---")
    # Test 1: Ma trận chéo trội
    A_gs1 = np.array([[10, 1, 1], [2, 10, 1], [2, 2, 10]], dtype=float)
    b_gs1 = np.array([12, 13, 14], dtype=float)
    verify_solver(solve_gauss_seidel, A_gs1, b_gs1, "GS diag dominant 3×3")

    # Test 2
    A_gs2 = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 4]], dtype=float)
    b_gs2 = np.array([1, 2, 3, 4], dtype=float)
    verify_solver(solve_gauss_seidel, A_gs2, b_gs2, "GS tridiagonal 4×4")

    # Test 3: Ma trận đường chéo
    verify_solver(solve_gauss_seidel, 5 * np.eye(3), np.array([5, 10, 15.0]), "GS diagonal")

    # Test 4: Ma trận SPD
    M = np.random.randn(5, 5)
    A_spd = M @ M.T + 10 * np.eye(5)
    b_spd = np.random.randn(5)
    verify_solver(solve_gauss_seidel, A_spd, b_spd, "GS SPD 5×5")

    # Test 5: Ma trận lớn chéo trội
    n = 20
    A_big = np.random.randn(n, n)
    A_big = A_big + n * np.eye(n)  # Chéo trội
    b_big = np.random.randn(n)
    verify_solver(solve_gauss_seidel, A_big, b_big, "GS diag dominant 20×20")

    # --- Condition Number ---
    print("\n--- Condition Number ---")
    print(f"  κ₂(I)          = {condition_number(np.eye(3)):.2f}")
    print(f"  κ₂(diag)       = {condition_number(np.diag([1, 2, 3])):.2f}")

    from scipy.linalg import hilbert
    for n in [3, 5, 10]:
        H = hilbert(n)
        kappa = condition_number(H)
        kappa_np = np.linalg.cond(H)
        print(f"  κ₂(Hilbert {n}×{n}) = {kappa:.2e} (NumPy: {kappa_np:.2e})")

    print("\n" + "=" * 60)
    print("  HOÀN TẤT TEST SUITE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
