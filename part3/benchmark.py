"""
benchmark.py — Thực nghiệm thời gian và phân tích ổn định số
Gồm:
  1. Benchmark thời gian: n ∈ {50, 100, 200, 500, 1000}
  2. Đồ thị log-log: thời gian vs n, so sánh O(n³)
  3. Phân tích ổn định: Hilbert (ill-conditioned) vs SPD (well-conditioned)
  4. Bảng số liệu tổng hợp
"""

import numpy as np
import time
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Backend không cần GUI
import matplotlib.pyplot as plt
from scipy.linalg import hilbert

# Import solvers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solvers import solve_gauss, solve_cholesky, solve_gauss_seidel, condition_number


# ===========================================================================
# 1. Utility Functions
# ===========================================================================

def generate_random_spd(n, cond_target=None):
    """
    Tạo ma trận ngẫu nhiên SPD (Symmetric Positive Definite).
    Nếu cond_target != None, điều chỉnh condition number.
    """
    M = np.random.randn(n, n)
    A = M @ M.T + n * np.eye(n)  # Đảm bảo SPD
    return A


def generate_diag_dominant(n):
    """Tạo ma trận chéo trội chặt hàng (để Gauss-Seidel hội tụ)."""
    A = np.random.randn(n, n)
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1.0
    return A


def relative_residual(A, x, b):
    """Tính sai số tương đối: ‖Ax̂ − b‖₂ / ‖b‖₂."""
    return np.linalg.norm(A @ x - b) / (np.linalg.norm(b) + 1e-16)


def timeit(func, *args, n_runs=5):
    """Đo thời gian trung bình qua n_runs lần chạy."""
    times = []
    result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times), result


# ===========================================================================
# 2. Benchmark Thời Gian
# ===========================================================================

def run_time_benchmark(sizes=None, n_runs=5):
    """
    Benchmark thời gian thực thi cho 3 phương pháp.
    
    Input:  sizes: danh sách kích thước n
            n_runs: số lần chạy trung bình
    Output: results dict
    """
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000]

    results = {
        'sizes': sizes,
        'gauss': {'times': [], 'stds': [], 'residuals': []},
        'cholesky': {'times': [], 'stds': [], 'residuals': []},
        'gauss_seidel': {'times': [], 'stds': [], 'residuals': []},
    }

    print("=" * 70)
    print("  BENCHMARK THỜI GIAN THỰC THI")
    print("=" * 70)
    print(f"  Kích thước: {sizes}")
    print(f"  Số lần chạy: {n_runs}")
    print("-" * 70)

    for n in sizes:
        print(f"\n  n = {n}:")

        # Tạo ma trận ngẫu nhiên MỘT LẦN cho mỗi n
        np.random.seed(42 + n)
        A = generate_random_spd(n)  # SPD để đảm bảo GS hội tụ và Cholesky hoạt động
        x_true = np.random.randn(n)
        b = A @ x_true

        # --- Gauss ---
        try:
            t_mean, t_std, x_gauss = timeit(solve_gauss, A, b, n_runs=n_runs)
            res = relative_residual(A, x_gauss, b)
            results['gauss']['times'].append(t_mean)
            results['gauss']['stds'].append(t_std)
            results['gauss']['residuals'].append(res)
            print(f"    Gauss:         {t_mean:.4f}s ± {t_std:.4f}s  |  residual = {res:.2e}")
        except Exception as e:
            print(f"    Gauss: FAILED ({e})")
            results['gauss']['times'].append(float('nan'))
            results['gauss']['stds'].append(float('nan'))
            results['gauss']['residuals'].append(float('nan'))

        # --- Cholesky ---
        try:
            t_mean, t_std, x_chol = timeit(solve_cholesky, A, b, n_runs=n_runs)
            res = relative_residual(A, x_chol, b)
            results['cholesky']['times'].append(t_mean)
            results['cholesky']['stds'].append(t_std)
            results['cholesky']['residuals'].append(res)
            print(f"    Cholesky:      {t_mean:.4f}s ± {t_std:.4f}s  |  residual = {res:.2e}")
        except Exception as e:
            print(f"    Cholesky: FAILED ({e})")
            results['cholesky']['times'].append(float('nan'))
            results['cholesky']['stds'].append(float('nan'))
            results['cholesky']['residuals'].append(float('nan'))

        # --- Gauss-Seidel ---
        try:
            def gs_wrapper(A, b):
                x, iters, _ = solve_gauss_seidel(A, b, tol=1e-10, max_iter=10000)
                return x
            t_mean, t_std, x_gs = timeit(gs_wrapper, A, b, n_runs=n_runs)
            res = relative_residual(A, x_gs, b)
            results['gauss_seidel']['times'].append(t_mean)
            results['gauss_seidel']['stds'].append(t_std)
            results['gauss_seidel']['residuals'].append(res)
            print(f"    Gauss-Seidel:  {t_mean:.4f}s ± {t_std:.4f}s  |  residual = {res:.2e}")
        except Exception as e:
            print(f"    Gauss-Seidel: FAILED ({e})")
            results['gauss_seidel']['times'].append(float('nan'))
            results['gauss_seidel']['stds'].append(float('nan'))
            results['gauss_seidel']['residuals'].append(float('nan'))

    return results


# ===========================================================================
# 3. Đồ Thị Log-Log
# ===========================================================================

def plot_loglog(results, save_path="benchmark_loglog.png"):
    """
    Vẽ đồ thị log-log: thời gian vs n.
    So sánh với đường lý thuyết O(n³).
    """
    sizes = np.array(results['sizes'], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Thời gian ---
    colors = {'gauss': '#e74c3c', 'cholesky': '#3498db', 'gauss_seidel': '#2ecc71'}
    labels = {'gauss': 'Gauss Elimination', 'cholesky': 'Cholesky Decomposition', 'gauss_seidel': 'Gauss-Seidel'}
    markers = {'gauss': 'o', 'cholesky': 's', 'gauss_seidel': '^'}

    for method in ['gauss', 'cholesky', 'gauss_seidel']:
        times = np.array(results[method]['times'])
        mask = ~np.isnan(times)
        if mask.any():
            ax1.loglog(
                sizes[mask], times[mask],
                marker=markers[method], color=colors[method],
                label=labels[method], linewidth=2, markersize=8,
            )

    # Đường O(n³) tham chiếu
    ref_times = results['gauss']['times']
    valid_times = [t for t in ref_times if not np.isnan(t)]
    if valid_times:
        c = valid_times[0] / (sizes[0] ** 3)
        ax1.loglog(
            sizes, c * sizes ** 3,
            '--', color='grey', alpha=0.7, linewidth=1.5,
            label=r'$O(n^3)$ reference',
        )

    ax1.set_xlabel('Kích thước ma trận n', fontsize=12)
    ax1.set_ylabel('Thời gian (giây)', fontsize=12)
    ax1.set_title('Benchmark: Thời gian vs Kích thước', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Sai số ---
    for method in ['gauss', 'cholesky', 'gauss_seidel']:
        residuals = np.array(results[method]['residuals'])
        mask = ~np.isnan(residuals)
        if mask.any():
            ax2.semilogy(
                sizes[mask], residuals[mask],
                marker=markers[method], color=colors[method],
                label=labels[method], linewidth=2, markersize=8,
            )

    ax2.set_xlabel('Kích thước ma trận n', fontsize=12)
    ax2.set_ylabel('Sai số tương đối ‖Ax̂−b‖/‖b‖', fontsize=12)
    ax2.set_title('Sai Số Tương Đối vs Kích Thước', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  📊 Đồ thị đã lưu: {save_path}")
    plt.close()


# ===========================================================================
# 4. Phân Tích Ổn Định Số
# ===========================================================================

def stability_analysis(save_path="stability_analysis.png"):
    """
    So sánh ổn định số giữa:
      - Ma trận Hilbert (ill-conditioned)
      - Ma trận ngẫu nhiên SPD (well-conditioned)
    """
    print("\n" + "=" * 70)
    print("  PHÂN TÍCH ỔN ĐỊNH SỐ")
    print("=" * 70)

    sizes = [3, 5, 7, 10, 12, 15]

    hilbert_data = {'sizes': sizes, 'cond': [], 'err_gauss': [], 'err_cholesky': []}
    spd_data = {'sizes': sizes, 'cond': [], 'err_gauss': [], 'err_cholesky': []}

    for n in sizes:
        print(f"\n  n = {n}:")
        np.random.seed(42)

        # --- Hilbert Matrix ---
        H = hilbert(n)
        x_true = np.ones(n)
        b_h = H @ x_true

        kappa_h = condition_number(H)
        hilbert_data['cond'].append(kappa_h)

        try:
            x_gauss = solve_gauss(H, b_h)
            err_g = np.linalg.norm(x_gauss - x_true) / np.linalg.norm(x_true)
        except Exception:
            err_g = float('nan')
        hilbert_data['err_gauss'].append(err_g)

        try:
            x_chol = solve_cholesky(H, b_h)
            err_c = np.linalg.norm(x_chol - x_true) / np.linalg.norm(x_true)
        except Exception:
            err_c = float('nan')
        hilbert_data['err_cholesky'].append(err_c)

        print(f"    Hilbert:  κ = {kappa_h:.2e}  |  err_Gauss = {err_g:.2e}  |  err_Chol = {err_c:.2e}")

        # --- Random SPD ---
        A_spd = generate_random_spd(n)
        b_spd = A_spd @ x_true

        kappa_s = condition_number(A_spd)
        spd_data['cond'].append(kappa_s)

        try:
            x_gauss = solve_gauss(A_spd, b_spd)
            err_g = np.linalg.norm(x_gauss - x_true) / np.linalg.norm(x_true)
        except Exception:
            err_g = float('nan')
        spd_data['err_gauss'].append(err_g)

        try:
            x_chol = solve_cholesky(A_spd, b_spd)
            err_c = np.linalg.norm(x_chol - x_true) / np.linalg.norm(x_true)
        except Exception:
            err_c = float('nan')
        spd_data['err_cholesky'].append(err_c)

        print(f"    SPD:      κ = {kappa_s:.2e}  |  err_Gauss = {err_g:.2e}  |  err_Chol = {err_c:.2e}")

    # --- Vẽ đồ thị ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Condition Number
    axes[0].semilogy(sizes, hilbert_data['cond'], 'r-o', label='Hilbert (ill-cond)', linewidth=2)
    axes[0].semilogy(sizes, spd_data['cond'], 'b-s', label='Random SPD (well-cond)', linewidth=2)
    axes[0].set_xlabel('Kích thước n')
    axes[0].set_ylabel('κ₂(A)')
    axes[0].set_title('Số Điều Kiện κ₂(A)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Sai số Gauss
    axes[1].semilogy(sizes, hilbert_data['err_gauss'], 'r-o', label='Hilbert', linewidth=2)
    axes[1].semilogy(sizes, spd_data['err_gauss'], 'b-s', label='Random SPD', linewidth=2)
    axes[1].set_xlabel('Kích thước n')
    axes[1].set_ylabel('Sai số tương đối')
    axes[1].set_title('Sai Số Gauss Elimination', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Sai số Cholesky
    axes[2].semilogy(sizes, hilbert_data['err_cholesky'], 'r-o', label='Hilbert', linewidth=2)
    axes[2].semilogy(sizes, spd_data['err_cholesky'], 'b-s', label='Random SPD', linewidth=2)
    axes[2].set_xlabel('Kích thước n')
    axes[2].set_ylabel('Sai số tương đối')
    axes[2].set_title('Sai Số Cholesky Decomposition', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phân tích Ổn Định Số: Hilbert vs SPD', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  📊 Đồ thị ổn định đã lưu: {save_path}")
    plt.close()

    return hilbert_data, spd_data


# ===========================================================================
# 5. Print Summary Table
# ===========================================================================

def print_summary_table(results):
    """In bảng tổng hợp kết quả benchmark."""
    print("\n" + "=" * 85)
    print("  BẢNG TỔNG HỢP KẾT QUẢ")
    print("=" * 85)

    header = f"{'n':>6} | {'Gauss (s)':>12} | {'Chol (s)':>12} | {'GS (s)':>12} | {'Res_G':>10} | {'Res_Chol':>10} | {'Res_GS':>10}"
    print(header)
    print("-" * 85)

    for i, n in enumerate(results['sizes']):
        tg = results['gauss']['times'][i]
        tc = results['cholesky']['times'][i]
        tgs = results['gauss_seidel']['times'][i]
        rg = results['gauss']['residuals'][i]
        rc = results['cholesky']['residuals'][i]
        rgs = results['gauss_seidel']['residuals'][i]

        print(f"{n:>6} | {tg:>12.4f} | {tc:>12.4f} | {tgs:>12.4f} | {rg:>10.2e} | {rc:>10.2e} | {rgs:>10.2e}")

    print("=" * 85)


# ===========================================================================
# 6. Main
# ===========================================================================

def main():
    """Chạy toàn bộ benchmark và phân tích."""
    # Đường dẫn lưu file
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Benchmark thời gian
    results = run_time_benchmark(sizes=[50, 100, 200, 500, 1000], n_runs=5)

    # Bảng tổng hợp
    print_summary_table(results)

    # Đồ thị log-log
    plot_loglog(results, save_path=os.path.join(output_dir, "benchmark_loglog.png"))

    # Phân tích ổn định
    stability_analysis(save_path=os.path.join(output_dir, "stability_analysis.png"))

    print("\n" + "=" * 70)
    print("  ✅ HOÀN TẤT BENCHMARK VÀ PHÂN TÍCH")
    print("=" * 70)


if __name__ == "__main__":
    main()
