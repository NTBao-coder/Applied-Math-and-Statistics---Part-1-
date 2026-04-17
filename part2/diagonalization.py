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

    """

    - Nếu n < 5: Sử dụng thuật toán QR tự cài đặt.

    - Nếu n >= 5: Sử dụng NumPy để tối ưu hiệu năng và độ chính xác.

    """
    n = len(A)
    if n >= 5:
        # Sử dụng NumPy cho bậc >= 5
        eigenvalues_np, eigenvectors_np = np.linalg.eig(np.array(A))
        return eigenvalues_np.tolist(), eigenvectors_np.tolist()
    else:
        # Thuật toán QR lặp cho bậc < 5
        V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        Ak = [row[:] for row in A]
        for _ in range(iterations):
            Q, R = qr_factorization_gram_schmidt(Ak) 
            Ak = matrix_multiply(R, Q)
            V = matrix_multiply(V, Q)
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
def calculate_determinant(matrix):
    """
    Tính định thức của ma trận vuông (Determinant) bằng phương pháp khử Gauss.
    Độ phức tạp: O(n^3)
    """
    n = len(matrix)
    # Sao chép ma trận sang danh sách mới để bảo toàn dữ liệu gốc
    a = [row[:] for row in matrix]
    det = 1.0

    for i in range(n):
        # Chọn phần tử trụ (Pivoting) để tăng độ chính xác số thực
        pivot_row = i
        for j in range(i + 1, n):
            if abs(a[j][i]) > abs(a[pivot_row][i]):
                pivot_row = j
        
        # Đổi chỗ hàng nếu phần tử trụ không nằm trên đường chéo chính
        if pivot_row != i:
            a[i], a[pivot_row] = a[pivot_row], a[i]
            det *= -1
            
        # Nếu phần tử trên đường chéo chính xấp xỉ 0, định thức bằng 0
        if abs(a[i][i]) < 1e-15:
            return 0.0
        
        det *= a[i][i]
        
        # Khử các hàng bên dưới cột i
        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]
            # Tối ưu: Chỉ chạy từ cột i+1 trở đi
            for k in range(i + 1, n):
                a[j][k] -= factor * a[i][k]
                
    return det

def diagonalize_matrix(A):
    """Chéo hóa ma trận vuông: A = P * D * P^-1 """
    n = len(A)
     # Kiểm tra ma trận vuông
    if n != len(A[0]):
        return None, None, None, "Lỗi: Chỉ chéo hóa ma trận vuông"

    lambdas, P = eigen_qr_iteration(A)
    
    # Kiểm tra tính chéo hóa bằng định thức 
    det_P = calculate_determinant(P)
    # Nếu định thức của ma trận vector riêng bằng 0 -> Không độc lập tuyến tính
    if abs(det_P) < 1e-10: 
        return None, None, None, "Thông báo: Hệ vector riêng không độc lập tuyến tính. Ma trận không chéo hóa được."

    D = create_diagonal_matrix(lambdas)
    
    # Vì P từ QR là ma trận trực giao nên P_inv = P^T)
    P_inv = matrix_transpose(P) 
    
    return P, D, P_inv, "Thành công"

def test_diagonalization(A):
    print("\n--- KIỂM CHỨNG CHÉO HÓA ---")
    
    # Nhận đủ 4 tham số trả về
    result = diagonalize_matrix(A)
    
    # Kiểm tra nếu hàm trả về lỗi (Lỗi ma trận không vuông hoặc không chéo hóa được)
    if result[0] is None:
        print(result[-1]) # In thông báo lỗi
        return

    P, D, P_inv, status = result
    
    # Ép kiểu để sử dụng toán tử @ của NumPy
    P_np, D_np, P_inv_np = np.array(P), np.array(D), np.array(P_inv)
    A_reconstructed = P_np @ D_np @ P_inv_np
    
    # 2. Kết quả từ thư viện NumPy 
    evals_np, evecs_np = np.linalg.eig(A)
    
    # 3. Tính toán sai số
    error = np.linalg.norm(np.array(A) - A_reconstructed) 
    
    print(f"Trạng thái: {status}")
    print(f"Giá trị riêng tự tính:\n{np.diag(D_np)}")
    print(f"Giá trị riêng từ NumPy:\n{evals_np}")
    print(f"Sai số tái tạo (||A - P*D*P_inv||): {error:.2e}")
    
    if error < 1e-8: # Để ngưỡng 1e-8 vì sai số số thực có thể tích lũy
        print("=> KẾT QUẢ CHÍNH XÁC.")
    else:
        print("=> CÓ SAI SỐ ĐÁNG KỂ (Có thể do ma trận không đối xứng nên P_inv != P^T).")
