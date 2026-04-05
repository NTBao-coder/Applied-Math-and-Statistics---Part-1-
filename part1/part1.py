import math
import numpy as np

# Hàm khử Gauss (trả về ma trận tam giác trên)
def gaussian_elimination(A, b):
    n, m = len(A), len(A[0])
    # Chuẩn hóa b sang dạng list of lists
    # Nếu giải hệ phương trình, b là vế phải của hệ và là 1 list
    # Nếu b là 1 ma trận (ví dụ dùng trong ma trận nghịch đảo) -> b là list của list
    if b is not None:
        b_norm = [[v] if not isinstance(v, list) else v[:] for v in b]
    else:
        b_norm = [[] for _ in range(n)]
    
    M = [A[i] + b_norm[i] for i in range(n)] # Ghép lại thành một ma trận tăng cường (A|b)
    m_total = len(M[0])
    s = 0 # Số lần thay đổi dòng
    pivot_cols = [] # Danh sách cột chứa pivot
    pivot_row = 0 # Dòng chứa pivot

    for k in range(m): # Duyệt qua từng cột
        if pivot_row >= n: break
        
        # Tìm dòng chứa phần tử lớn nhất làm pivot 
        p = pivot_row
        for i in range(pivot_row + 1, n):
            if abs(M[i][k]) > abs(M[p][k]):
                p = i
        
        # Xử lý ngoại lệ: Nếu tất cả các phần tử trong cột đều là 0 -> Phương trình có thể không có nghiệm duy nhất
        if abs(M[p][k]) < 1e-12: continue
        
        # Thêm cột có chứa pivot vào list
        pivot_cols.append(k)
        # Hoán đổi dòng chứa pivot với dòng hiện tại
        M[pivot_row], M[p] = M[p], M[pivot_row]
        # Tăng số lần đếm hoán đổi dòng lên 1
        if p != pivot_row: s += 1
        
        # Khử Gauss
        for i in range(pivot_row + 1, n):
            factor = M[i][k] / M[pivot_row][k]
            M[i][k] = 0 # Triệt tiêu chính xác về 0
            for j in range(k + 1, m_total):
                M[i][j] -= factor * M[pivot_row][j]
        pivot_row += 1
                
    return M, s, pivot_cols

# Hàm khử các phần tử trên đường chéo
def backward_elimination(M, pivot_cols):
    A_rref = [row[:] for row in M]
    n_rows = len(A_rref)
    # Khử ngược từ dưới lên dựa trên danh sách các dòng chứa pivot
    for i in range(len(pivot_cols) - 1, -1, -1):
        r, c = i, pivot_cols[i] # Dòng r, cột c chứa pivot
        pivot_val = A_rref[r][c]
        A_rref[r] = [x / pivot_val for x in A_rref[r]]
        
        for k in range(r - 1, -1, -1):
            factor = A_rref[k][c]
            for j in range(c, len(A_rref[0])):
                A_rref[k][j] -= factor * A_rref[r][j]
    return A_rref

# Hàm giải hệ phương trình
def solver(A, b):
    n, m = len(A), len(A[0])
    M, s, pivot_cols = gaussian_elimination(A, b)
    
    # Kiểm tra vô nghiệm
    for i in range(n):
        if all(abs(M[i][j]) < 1e-12 for j in range(m)):
            if abs(M[i][m]) > 1e-12:
                return "Hệ phương trình vô nghiệm", M, None, s

    if len(pivot_cols) == m:
        # Nghiệm duy nhất
        M_rref = backward_elimination(M, pivot_cols)
        x = [row[m] for row in M_rref[:m]]
        return "Hệ phương trình có nghiệm duy nhất", M, x, s
    else:
        # Vô số nghiệm
        M_rref = backward_elimination(M, pivot_cols)
        sol = solve_infinite_solutions(M_rref, pivot_cols, m)
        return "Hệ phương trình có vô số nghiệm", M_rref, sol, s

# Hàm giải hệ vô số nghiệm
def solve_infinite_solutions(M_rref, pivot_cols, m_vars, epsilon=1e-12):
    n_rows = len(M_rref)
    
    # 1. Xác định các ẩn tự do (cột không có pivot)
    free_vars = [j for j in range(m_vars) if j not in pivot_cols]
    
    # 2. Tìm nghiệm riêng Xp (Đặt tất cả ẩn tự do = 0)
    # Với RREF, giá trị các ẩn chính (pivot) chính là cột b tương ứng
    x_particular = [0.0] * m_vars
    for i, p_col in enumerate(pivot_cols):
        if i < n_rows:
            x_particular[p_col] = M_rref[i][m_vars]
            
    # 3. Tìm hệ nghiệm cơ bản
    # Mỗi ẩn tự do sẽ tạo ra một vector trong cơ sở
    basis_vectors = []
    for f_var in free_vars:
        v = [0.0] * m_vars
        v[f_var] = 1.0  # Đặt ẩn tự do này bằng 1, các ẩn tự do khác bằng 0
        
        for i, p_col in enumerate(pivot_cols):
            if i < n_rows:
                # Chuyển vế: x_pivot + coef * x_free = 0 => x_pivot = -coef
                v[p_col] = -M_rref[i][f_var]
        
        basis_vectors.append(v)
        
    return {
        "particular": x_particular, # Nghiệm riêng
        "basis": basis_vectors, # Nghiệm cơ bản
        "free_indices": free_vars # Ẩn tự do
    }

# Hàm giải hệ phương trình có nghiệm duy nhất
def back_substitution(U, c):
    n = len(U)
    x = [0] * n # Khởi tạo nghiệm ban đầu là 0
    
    for i in range(n - 1, -1, -1): # Tìm nghiệm của từng biến, duyệt từ trên xuống dưới
        if abs(U[i][i]) < 1e-12:
            continue
        
        sum = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (c[i] - sum) / U[i][i]
        
    return x

# Hàm tính định thức
def determinant(A):
    A_temp = [row[:] for row in A]
    
    # Khử Gauss
    M, s, pivot_cols = gaussian_elimination(A_temp, b = None)
    
    # Nếu hạng ma trận < số dòng của ma trận -> Định thức bằng 0
    if(len(pivot_cols) < len(A)):
        return 0
    
    det = 1
    # Tính tích đường chéo
    for i in range(len(A)):
        det *= M[i][i]
    
    # Đổi dấu nếu số lần đổi dòng lẻ
    if(s % 2 == 1):
        det = -det
    
    return det

# Hàm tìm ma trận nghịch đảo
def inverse(A):
    n = len(A)
    # Tạo ma trận đơn vị I
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # Khử xuôi -> Ma trận tam giác trên
    M_ref, s, pivot_cols = gaussian_elimination(A, I)
    
    # Kiểm tra xem A có thể đưa về thành ma trận đơn vị không
    if(len(pivot_cols) < n):
        return "Ma trận không có nghịch đảo"

    # Khử ngược -> Ma trận đơn vị
    M_rref = backward_elimination(M_ref, pivot_cols)
    
    # Lấy ma trận bên phải
    A_inverse = [row[n:] for row in M_rref]
    return A_inverse

# Hạng và cơ sở
def rank_and_basis(A):
    n_rows = len(A)
    n_cols = len(A[0])
    b_zero = [0.0] * n_rows
    
    # Khử Gauss cho ma trận
    M_ref, s, pivot_cols = gaussian_elimination(A, b_zero)
    
    rank = len(pivot_cols) # Hạng của ma trận   
    
    # 1. Cơ sở dòng
    # Lấy tất cả các dòng khác 0 
    row_basis = [row[:n_cols] for row in M_ref if any(abs(x) > 1e-12 for x in row[:n_cols])]

    # 2. Cơ sở cột
    # Lấy tất cả các cột (của ma trận A ban đầu) có chứa pivot 
    col_basis = [[A[i][j] for i in range(n_rows)] for j in pivot_cols]
    
    # 3. Cơ sở nghiệm
    M_rref = backward_elimination(M_ref, pivot_cols)
    solution = solve_infinite_solutions(M_rref, pivot_cols, n_cols, epsilon = 1e-12)
    
    return{
        "rank" : rank, # Hạng của ma trận
        "row_basis" : row_basis,  # Cơ sở dòng
        "col_basis" : col_basis, # Cơ sở cột
        "null_basis" : solution # Cơ sở không gian nghiệm
    }
    
# Hàm kiểm tra lại xem hệ nghiệm có thỏa hệ phương trình hay không
def verify_solution(A, x, b):
    
    # Chuyển ma trận và vector sang Numpy và cho kiểu dữ liệu là số thực để tính toán chính xác
    A_np = np.array(A, dtype=float)
    x_np = np.array(x, dtype=float)
    b_np = np.array(b, dtype=float).flatten()
    
    # Đặt sai số cho phép tính vì tính toán trong máy sẽ có sai số
    EPSILON = 1e-6
    
    # tính A.x với toán tử nhân @ trong Numpy
    Ax = A_np @ x_np
    
    # Trả về phép so sánh A.x = b với sai số là 10^-6
    return np.allclose(Ax, b_np, atol=EPSILON)

# Hàm biểu diễn testcase
def print_test_case(name, A, b, result_tuple):
    status, M, res, s = result_tuple
    n = len(A)
    m = len(A[0])
    
    print(f"\n{'='*15} {name} {'='*15}")
    print(f"Trạng thái: {status}")

    # 1. Định thức (Chỉ cho ma trận vuông)
    if n == m:
        det_val = determinant(A)
        print(f"Định thức (det A): {round(det_val, 4)}")
        
        # 2. Ma trận nghịch đảo (Chỉ khi det != 0)
        if abs(det_val) > 1e-12:
            inv_result = inverse(A)
            if isinstance(inv_result, list):
                print("Ma trận nghịch đảo (A^-1):")
                for row in inv_result:
                    print(f"  {[round(val, 4) for val in row]}")
            else:
                print(f"Nghịch đảo: {inv_result}")
        else:
            print("Nghịch đảo: Không tồn tại")
    else:
        print("Định thức/Nghịch đảo: Không xác định")

    # 3. Hiển thị nghiệm của hệ
    if status == "Hệ phương trình có nghiệm duy nhất":
        print(f"Nghiệm x: {[round(val, 4) for val in res]}")
    elif status == "Hệ phương trình có vô số nghiệm":
        print(f"Nghiệm riêng Xp: {[round(val, 4) for val in res['particular']]}")
        print(f"Hệ nghiệm cơ bản: {[[round(x, 4) for x in v] for v in res['basis']]}")
    
    # Hạng và ma trận    
    result_rb = rank_and_basis(A)
    rank = result_rb["rank"]
    basis = result_rb["null_basis"]
    print(f"Hạng của ma trận: {rank}")
    
    basis_vector = basis['basis']
    if basis_vector and len(basis_vector) > 0:
        print("Cơ sở không gian nghiệm (Null Space Basis):")
        for i, v in enumerate(basis_vector):
            # v lúc này là một list các số thực, ví dụ [1.0, 0.5, 0.0]
            print(f"  v{i+1}: {[round(float(x), 4) for x in v]}")
    else:
        print("Cơ sở không gian nghiệm: {0} (Chỉ có nghiệm tầm thường)")
        
    # Kiểm tra kết quả bằng Numpy
    if res != None:
        target_x = res['particular'] if isinstance(res, dict) else res
        
        is_valid = verify_solution(A, target_x, b)
        if(is_valid):
            print("Kết quả trùng với kiểm chứng Numpy")
        else:
            print("Kết quả sai lệch với kiểm chứng Numpy")    
    
    print("-" * 50)

# Hàm nhập ma trận
def input_matrix():
    print("\n------NHẬP MA TRẬN------")
    try:
        n = int(input("Nhập số hàng (n): "))
        m = int(input("Nhập số cột (m): "))
        
        A = []
        print(f"Nhập từng hàng của ma trận (Các phần tử cách nhau bởi dấu cách):")
        for i in range(n):
            while True:
                line = input(f"  Hàng {i+1}: ").split()
                if len(line) == m:
                    A.append([float(x) for x in line])
                    break
                print(f"Lỗi: Bạn phải nhập đúng {m} số. Vui lòng nhập lại hàng {i+1}.")
        
        print("\nNhập vector b (Các phần tử cách nhau bởi dấu cách):")
        while True:
            line = input("  b = ").split()
            if len(line) == n:
                b = [float(x) for x in line]
                break
            print(f"Lỗi: Vector b phải có {n} phần tử.")
            
        return A, b
    except ValueError:
        print("Lỗi: Vui lòng chỉ nhập các chữ số.")
        return None, None
    
# Hàm main (chạy testcase)
if __name__ == "__main__":
    A, b = input_matrix()
    result = solver(A, b)
    print_test_case("Kết quả", A, b, result)