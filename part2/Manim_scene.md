# Kịch Bản Trực Quan Hóa Manim - Đồ Án 1 (Phần 2)

**Chủ đề:** Phân rã Cholesky & Chéo hóa ma trận
[cite_start]**Thuật toán sử dụng:** Phân rã Cholesky ($A=LL^T$)[cite: 175], thuật toán QR lặp (Gram-Schmidt) cho chéo hóa.

## 1. Yêu Cầu Kỹ Thuật (Guidelines)
* [cite_start]**Định dạng video:** `.mp4`[cite: 310].
* [cite_start]**Chất lượng:** Độ phân giải tối thiểu 720p[cite: 310].
* [cite_start]**Thời lượng:** Từ 2 đến 30 phút[cite: 191, 310].
* **Lưu ý hệ điều hành:** Đảm bảo cài đặt đủ `ffmpeg`, `sox`, và `cairo` (đặc biệt quan trọng trên macOS/Apple Silicon).

---

## 2. Phân Cảnh Chi Tiết (Scenes)

### Scene 1: Giới thiệu Bài Toán (Introduction)
* **Mục tiêu:** Giới thiệu ma trận đầu vào và xác nhận điều kiện áp dụng thuật toán.
* **Kịch bản hoạt ảnh:**
  1. Hiển thị tiêu đề chính giữa màn hình: "Phân Rã Cholesky & Chéo Hóa Ma Trận".
  2. Đưa ma trận test $A$ ra giữa màn hình: 
     $A = \begin{bmatrix} 4 & 2 \\ 2 & 10 \end{bmatrix}$
  3. Chạy hiệu ứng xuất hiện các dòng text kiểm tra điều kiện (mô phỏng logic của code): 
     * "Kiểm tra tính đối xứng... OK"
     * "Kiểm tra xác định dương... OK"
  4. Thu nhỏ và đưa ma trận $A$ sang góc để nhường không gian cho Scene tiếp theo.

### Scene 2: Quá Trình Phân Rã Cholesky ($A = LL^T$)
* [cite_start]**Mục tiêu:** Trực quan hóa từng bước tính toán ma trận tam giác dưới $L$[cite: 175].
* **Kịch bản hoạt ảnh:**
  1. [cite_start]Hiện công thức tổng quát $A = LL^T$ [cite: 175] ở góc trên.
  2. Hiển thị ma trận $L$ dưới dạng các ẩn số: $L = \begin{bmatrix} L_{00} & 0 \\ L_{10} & L_{11} \end{bmatrix}$. 
  3. **Tính $L_{00}$:** Highlight phần tử $A_{00}$ (số 4). Hiển thị phép tính $L_{00} = \sqrt{4} = 2$. Cập nhật số 2 vào ma trận $L$.
  4. **Tính $L_{10}$:** Highlight $A_{10}$ (số 2). Hiển thị phép tính $L_{10} = \frac{2}{2} = 1$. Cập nhật số 1 vào ma trận $L$.
  5. **Tính $L_{11}$:** Highlight $A_{11}$ (số 10) và $L_{10}$ (số 1). Hiển thị phép tính $L_{11} = \sqrt{10 - 1^2} = 3$. Cập nhật số 3 vào ma trận $L$.
  6. **Kiểm chứng:** Trực quan hóa phép nhân $L \times L^T$ để chứng minh kết quả trả về đúng ma trận $A$ ban đầu.

### Scene 3: Chéo Hóa Ma Trận ($A = P D P^{-1}$)
* [cite_start]**Mục tiêu:** Hiển thị kết quả của thuật toán chéo hóa[cite: 198].
* **Kịch bản hoạt ảnh:**
  1. Xóa không gian làm việc của Scene 2. Hiện công thức chéo hóa: $A = P D P^{-1}$[cite: 133].
  2. Hiện thông báo ngắn: "Áp dụng thuật toán QR Iteration".
  3. Trình bày các ma trận kết quả (làm tròn để tối ưu hiển thị):
     * Ma trận đường chéo $D \approx \begin{bmatrix} 3.39 & 0 \\ 0 & 10.61 \end{bmatrix}$ (chứa giá trị riêng).
     * Ma trận trực giao $P \approx \begin{bmatrix} -0.96 & 0.29 \\ 0.29 & 0.96 \end{bmatrix}$ (chứa vector riêng).
  4. Ghép hoàn chỉnh phương trình: $A \approx P \times D \times P^T$.
  5. Hiện dòng chữ "Hoàn thành chéo hóa!" màu xanh lá, sau đó làm mờ toàn bộ màn hình (FadeOut) để kết thúc video.