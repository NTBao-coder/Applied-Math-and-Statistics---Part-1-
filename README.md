# Applied-Math-and-Statistics---Part-1-
Bài Tập Lab 1 toán ứng dụng và thống kê. Đồ án 1: Ma Trận và Cơ Sở của Tính Toán Khoa Học. 
# 🚀 Matrix & Scientific Computing Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/Manim-Animation-green.svg">
  <img src="https://img.shields.io/badge/Status-Completed-success.svg">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

---

## 🧭 Overview

This project implements core algorithms in **Numerical Linear Algebra** from scratch — no shortcuts, no black-box libraries.

It focuses on:

* Understanding **how algorithms actually work**
* Evaluating **numerical stability**
* Measuring **real computational performance**

> This is not just a math project — it’s a **systems + algorithm thinking project**.

---

## 🔥 Highlights

* ✅ Full implementation of **Gaussian Elimination with Partial Pivoting**
* ✅ Matrix operations: determinant, inverse, rank, basis
* ✅ Advanced decomposition (LU / QR / SVD / Cholesky)
* ✅ 🎥 **Mathematical visualization using Manim**
* ✅ 📊 Benchmarking on large-scale matrices (n up to 1000)
* ✅ Numerical stability analysis (Hilbert vs SPD matrices)

---

## 🧠 Technical Depth

### 1. Gaussian Elimination (Core Engine)

* Partial Pivoting to reduce floating-point error
* Back-substitution for solving triangular systems
* Used as foundation for:

  * Determinant computation
  * Matrix inversion
  * Rank detection

### 2. Matrix Decomposition

Choose and implement one:

| Method   | Strength                      |
| -------- | ----------------------------- |
| LU       | Fast solving multiple systems |
| QR       | High numerical stability      |
| SVD      | Best for data science / PCA   |
| Cholesky | Optimized for SPD matrices    |

### 3. Numerical Stability

* Condition Number analysis
* Error propagation:

  * ‖Ax̂ - b‖ / ‖b‖
* Ill-conditioned vs well-conditioned systems

### 4. Performance Engineering

* Time complexity validation (O(n³))
* Log-log performance plots
* Benchmark across matrix sizes

---

## 🗂️ Project Structure

```bash
Group_<ID>/
│── README.md
│── requirements.txt
│
├── report/                 # Academic report (PDF + LaTeX)
├── part1/                  # Gaussian elimination & applications
├── part2/                  # Matrix decomposition + Manim
├── part3/                  # Benchmark & numerical analysis
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run core algorithm

```bash
python part1/gaussian.py
```

### Run decomposition

```bash
python part2/decomposition.py
```

### Run benchmark

```bash
python part3/benchmark.py
```

### Launch notebooks

```bash
jupyter notebook
```

---

## 🎥 Visualization (Manim)

This project includes animated explanations of:

* Gaussian elimination steps
* Vector orthogonalization (QR)
* Matrix transformation (SVD)

👉 Output:

```
part2/demo_video.mp4
```

> If you're learning linear algebra, this is where things “click”.

---

## 📊 Example Results

| Matrix Size | Time (s) | Relative Error |
| ----------- | -------- | -------------- |
| 50          | 0.01     | 1e-12          |
| 200         | 0.12     | 1e-10          |
| 1000        | 4.85     | 1e-7           |

📌 Observations:

* Time grows ~O(n³) as expected
* Error increases with matrix condition number

---

## ⚠️ Constraints (Important)

* ❌ No use of:

  * `numpy.linalg.solve`
  * `numpy.linalg.inv`
  * `scipy.linalg.*`
* ✅ Allowed:

  * NumPy **only for verification**

---

## 🧪 Testing Strategy

* At least **5 test cases per function**
* Edge cases:

  * Singular matrix
  * Near-zero pivot
  * Large condition number

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Manim** (animation engine)
* **NumPy** (verification only)
* **Matplotlib** (visualization)
* **Jupyter Notebook**

---

## 👥 Team

| Name      | Role                  |
| --------- | --------------------- |
| Your Name | Core Algorithms       |
| Member 2  | Visualization (Manim) |
| Member 3  | Benchmark & Analysis  |

---

## 📚 References

* Gilbert Strang — *Introduction to Linear Algebra*
* Golub & Van Loan — *Matrix Computations*
* Trefethen & Bau — *Numerical Linear Algebra*
* 3Blue1Brown — *Essence of Linear Algebra*

---

## 💡 What I Learned

* Writing algorithms from scratch exposes hidden complexity
* Numerical stability is often more important than correctness
* Visualization helps understand abstract math deeply
* Performance matters — theory ≠ reality

---

## 🌟 Future Improvements

* [ ] Add GPU acceleration (CUDA / PyTorch)
* [ ] Build interactive UI (Streamlit / Gradio)
* [ ] Extend to sparse matrices
* [ ] Integrate with ML pipelines (PCA via SVD)

---

## ⭐ Support

If this project helps you:

* ⭐ Star the repo
* 🍴 Fork it
* 🧠 Use it as a learning reference

---

## 📬 Contact

* Email: [your-email@example.com](mailto:your-email@example.com)
* GitHub: https://github.com/your-username

---

> “If you can implement it, you understand it.”
