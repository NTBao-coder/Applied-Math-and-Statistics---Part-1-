# 3. Phần 3: Giải Hệ Phương Trình và Phân Tích Hiệu Năng

---

> **Tóm tắt yêu cầu Phần 3:** So sánh các phương pháp giải $\mathbf{Ax} = \mathbf{b}$, bao gồm Gauss (Phần 1), phân rã (Phần 2) và ít nhất một phương pháp lặp. Phân tích tính ổn định số và chi phí tính toán qua thực nghiệm.

## 3.1. Lý Thuyết Sai Số và Tính Ổn Định Số

### 3.1.1. Số Điều Kiện (Condition Number)

**Định nghĩa 3.1 (Số điều kiện).** *Số điều kiện* của ma trận $\mathbf{A}$ đối với chuẩn $p$ là:

$$\kappa_p(\mathbf{A}) = \|\mathbf{A}\|_p \cdot \|\mathbf{A}^{-1}\|_p \tag{10}$$

Đối với chuẩn spectral (chuẩn 2): $\kappa_2(\mathbf{A}) = \frac{\sigma_{\max}(\mathbf{A})}{\sigma_{\min}(\mathbf{A})}$.

**Định lý 3.1 (Phân tích sai số nghiệm).** *Nếu $\mathbf{Ax} = \mathbf{b}$ là nghiệm đúng và $\mathbf{A\hat{x}} = \mathbf{b} + \delta\mathbf{b}$ là nghiệm tính với nhiễu dữ liệu, thì:*

$$\frac{\|\mathbf{\hat{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(\mathbf{A}) \cdot \frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|} \tag{11}$$

*Khi $\kappa(\mathbf{A})$ lớn, hệ bị điều kiện kém (ill-conditioned): sai số nhỏ trong dữ liệu đầu vào gây ra sai số lớn trong nghiệm.*

### 3.1.2. So Sánh Các Phương Pháp

| Phương pháp | Loại | Điều kiện áp dụng | Chi phí | Ổn định |
| :--- | :--- | :--- | :--- | :--- |
| Gauss (Partial Pivot) | Trực tiếp | Mọi $\mathbf{A}$ khả nghịch | $O(n^3)$ | Cao |
| LU với pivot | Trực tiếp | Mọi $\mathbf{A}$ khả nghịch | $O(n^3)$ | Cao |
| QR (Householder) | Trực tiếp | Mọi $\mathbf{A}$ (kể cả chữ nhật) | $O(n^3)$ | Rất cao |
| Cholesky | Trực tiếp | $\mathbf{A}$ đối xứng xác định dương | $O(n^3/6)$ | Rất cao |
| Gauss–Seidel | Lặp | Chéo trội hàng / SPD | $O(kn^2)$ | Trung bình |

### 3.1.3. Phương Pháp Lặp Gauss–Seidel

Phân rã $\mathbf{A} = \mathbf{D} + \mathbf{L} + \mathbf{U}$ (đường chéo $\mathbf{D}$, tam giác dưới thuần $\mathbf{L}$, tam giác trên thuần $\mathbf{U}$). Ta có công thức lặp:

$$\mathbf{x}^{(k+1)} = -(\mathbf{D} + \mathbf{L})^{-1}\mathbf{U}\mathbf{x}^{(k)} + (\mathbf{D} + \mathbf{L})^{-1}\mathbf{b} \tag{12}$$

Nếu ta viết theo từng thành phần thì công thức lặp sẽ có dạng như sau:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^n a_{ij} x_j^{(k)} \right), \quad i = 1, \dots, n. \tag{13}$$

**Điều kiện hội tụ:** $\mathbf{A}$ chéo trội chặt hàng, tức $|a_{ii}| > \sum_{j \ne i} |a_{ij}|$ với mọi $i$.

---

## 3.2. Yêu Cầu Cài Đặt và Phân Tích

> **Yêu cầu — Phần 3**
> 
> 1. **Cài đặt** ít nhất 3 phương pháp giải $\mathbf{Ax} = \mathbf{b}$ (bao gồm Gauss từ Phần 1 và phân rã từ Phần 2).
> 
> 2. **Thực nghiệm** với ma trận ngẫu nhiên kích thước $n \in \{50, 100, 200, 500, 1000\}$:
>    * Đo thời gian thực thi (trung bình 5 lần chạy).
>    * Đo sai số tương đối: $\|\mathbf{A\hat{x}} - \mathbf{b}\|_2 / \|\mathbf{b}\|_2$.
>    * Vẽ đồ thị $\log-\log$: thời gian vs $n$ và so sánh với đường lý thuyết $O(n^3)$.
> 
> 3. **Phân tích ổn định** với hai loại ma trận:
>    * *Ma trận Hilbert $H_n$* (số điều kiện rất lớn — ill-conditioned).
>    * *Ma trận ngẫu nhiên SPD* (số điều kiện nhỏ — well-conditioned).
> 
> 4. **Báo cáo:** Trình bày trong Jupyter Notebook với bảng số liệu, biểu đồ có chú thích đầy đủ.

## 3.3. Tiêu Chí Đánh Giá — Phần 3

| Tiêu chí | Mô tả | Điểm |
| :--- | :--- | :--- |
| **Cài đặt phương pháp lặp** | Đúng, có kiểm tra điều kiện hội tụ | 0.5 |
| **Thực nghiệm thời gian** | Đủ kích thước, có đồ thị $\log-\log$ | 0.5 |
| **Phân tích ổn định số** | Ma trận Hilbert vs ma trận ngẫu nhiên | 0.5 |
| **Nhận xét và kết luận** | Phân tích có chiều sâu, có số liệu | 0.25 |
| **Trình bày Notebook** | Rõ ràng, có visualization đầy đủ | 0.25 |
| **Tổng Phần 3** | | **2.0** |