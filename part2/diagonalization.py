import math
import numpy as np
"""
Description:
-qr_factorization_gram_schmidt(A):Triển khai phân rã ma trận A = QR
Sử dụng quá trình Gram-Schmidt cổ điển để tạo ra ma trận trực giao Q và ma trận tam giác trên R
-matrix_multiply: Hàm nhân hai ma trận A (n x m) và B (m x p)
-eigen_qr_iteration(A, iterations=500):Thuật toán lặp để tìm các trị riêng (eigenvalues) và vectơ riêng (eigenvectors)
matrix_transpose:Tính ma trận chuyển vị của P
def create_diagonal_matrix(lambdas):Tạo ma trận đường chéo D từ danh sách trị riêng
-diagonalize_matrix(A):Hàm tổng hợp để thực hiện chéo hóa hoàn chỉnh dưới dạng A = PDP^{-1}
-test_diagonalization(A):Hàm kiểm chứng độ chính xác của thuật toán.
So sánh trị riêng tự tính với hàm tiêu chuẩn np.linalg.eig của NumPy.
"""
def qr_factorization_gram_schmidt(A):
    """ cài đặt phân rã QR bằng Gram-Schmidt"""
    n = len(A)          
    m = len(A[0])

    Q = [[0.0 for _ in range(m)] for _ in range(n)]

    R = [[0.0 for _ in range(m)] for _ in range(m)]
    for k in range(m):
        v = [A[i][k] for i in range(n)]
        for i in range(k):
           
            dot_product = sum(Q[row][i] * A[row][k] for row in range(n))
            R[i][k] = dot_product
            for row in range(n):
                v[row] = v[row] - R[i][k] * Q[row][i]
        
       
        norm_v = math.sqrt(sum(x**2 for x in v))
        if norm_v > 1e-10:
            R[k][k] = norm_v
            for row in range(n):
                Q[row][k] = v[row] / R[k][k]
    return Q, R

def matrix_multiply(A, B):
    """Hàm nhân hai ma trận A (n x m) và B (m x p)"""
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    # Khởi tạo ma trận kết quả toàn số 0
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C

def eigen_qr_iteration(A, iterations=500):
    """Thuật toán QR lặp để tìm trị riêng và vector riêng """
    n = len(A)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    Ak = [row[:] for row in A]
    for _ in range(iterations):
        Q, R =  qr_factorization_gram_schmidt(Ak) 
        Ak = matrix_multiply(R, Q)
        V = matrix_multiply(V, Q)
    # trị riêng (là các phần tử trên đường chéo của Ak)
    eigenvalues = [Ak[i][i] for i in range(n)]    
    return eigenvalues, V

def matrix_transpose(P):
    """Tính ma trận chuyển vị của P"""
    n = len(P)
    m = len(P[0])
    # Tạo ma trận mới m hàng, n cột
    return [[P[j][i] for j in range(n)] for i in range(m)]

def create_diagonal_matrix(lambdas):
    """Tạo ma trận đường chéo D từ danh sách trị riêng"""
    n = len(lambdas)
    D = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        D[i][i] = lambdas[i]
    return D

def diagonalize_matrix(A):
    """Chéo hóa ma trận vuông: A = P * D * P^-1 """
    n = len(A)
    m = len(A[0])
    
    # Kiểm tra ma trận vuông
    if n != m:
        return "Lỗi: Chỉ chéo hóa ma trận vuông"

    # 1. Tìm D (giá trị riêng) và P (vector riêng) 
    lambdas, P = eigen_qr_iteration(A)
    D = create_diagonal_matrix(lambdas)
    
    # 2. Tính P_inv (Vì P từ QR là ma trận trực giao nên P_inv = P^T)
    P_inv = matrix_transpose(P)
    
    return P, D, P_inv

def test_diagonalization(A):
    print("--- KIỂM CHỨNG CHÉO HÓA ---")
    
    # 1. Kết quả từ hàm tự cài đặt 
    P, D, P_inv = diagonalize_matrix(A)
    A_reconstructed = P @ D@ P_inv
    
    # 2. Kết quả từ thư viện NumPy 

    evals_np, evecs_np = np.linalg.eig(A)
    
    # 3. Tính toán sai số (Error Analysis)
    # Độ lệch giữa ma trận gốc và ma trận tái tạo từ P, D, P_inv
    error = np.linalg.norm(A - A_reconstructed) 
    
    print(f"Ma trận D (Giá trị riêng tự tính):\n{np.diag(D)}")
    print(f"Giá trị riêng từ NumPy:\n{evals_np}")
    print(f"Sai số tái tạo (||A - P*D*P_inv||): {error:.2e}")
    
    if error < 1e-10:
        print("=> KẾT QUẢ CHÍNH XÁC.")
    else:
        print("=> CÓ SAI SỐ ĐÁNG KỂ .")
