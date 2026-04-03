import math
import numpy as np

def qr_factorization_gram_schmidt(A):
    """ cài đặt phân rã QR bằng Gram-Schmidt"""
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))
    for k in range(m):
        v = A[:, k].copy()
        for i in range(k):
           
            R[i, k] = np.dot(Q[:, i], A[:, k])
            v = v - R[i, k] * Q[:, i]
        
       
        norm_v = np.sqrt(np.sum(v**2))
        if norm_v > 1e-10:
            R[k, k] = norm_v
            Q[:, k] = v / R[k, k]
    return Q, R

def  eigen_qr_iteration(A, iterations=500):
    """Thuật toán QR lặp để tìm trị riêng và vector riêng """
    n = A.shape[0]
    V = np.eye(n)
    Ak = A.copy().astype(float)
    for _ in range(iterations):
        Q, R =  qr_factorization_gram_schmidt(Ak) 
        Ak = R @ Q
        V = V @ Q
    return np.diag(Ak), V
def diagonalize_matrix(A):
    """Chéo hóa ma trận vuông: A = P * D * P^-1 """
    if A.shape[0] != A.shape[1]:
        return "Lỗi: Chỉ chéo hóa ma trận vuông "

    # 1. Tìm D (giá trị riêng) và P (vector riêng) 
    lambdas, P = eigen_qr_iteration(A)
    D = np.diag(lambdas)
    
    # 2. Tính P_inv (Vì P từ QR là ma trận trực giao nên P_inv = P^T)
    P_inv = P.T 
    
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