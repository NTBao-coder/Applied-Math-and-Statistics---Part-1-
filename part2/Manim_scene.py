"""
Module: manim_scene.py
Description: Video trực quan hóa Phân rã Cholesky và Chéo hóa ma trận cho Đồ án 1.
Chạy file bằng lệnh (đặc biệt lưu ý môi trường macOS):
    manim -pql part2/Manim_scene.py CholeskyAndDiagonalization
"""

from manim import *

class CholeskyAndDiagonalization(Scene):
    def construct(self):
        # Thiết lập màu sắc và phông chữ
        Text.set_default(font="sans-serif")
        
        # ==========================================
        # SCENE 1: GIỚI THIỆU BÀI TOÁN
        # ==========================================
        title = Text("Phân Rã Cholesky & Chéo Hóa Ma Trận", font_size=40, color=BLUE).to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN))

        intro_text = Text("Xét ma trận đối xứng xác định dương (SPD):", font_size=28).next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(intro_text, shift=DOWN))

        # Khởi tạo ma trận A (lấy từ test_decomposition.py: A = [[4, 2], [2, 10]])
        A_matrix = Matrix([[4, 2], [2, 10]])
        A_label = MathTex("A = ")
        A_group = VGroup(A_label, A_matrix).arrange(RIGHT).next_to(intro_text, DOWN, buff=0.8)

        self.play(Write(A_group))
        self.wait(1)

        # Mô phỏng quá trình kiểm tra Edge Cases
        check_sym = Text("Kiểm tra tính đối xứng... OK", font_size=24, color=GREEN)
        check_spd = Text("Kiểm tra xác định dương... OK", font_size=24, color=GREEN)
        
        checks = VGroup(check_sym, check_spd).arrange(DOWN, aligned_edge=LEFT).next_to(A_group, DOWN, buff=1)

        self.play(FadeIn(check_sym, shift=RIGHT))
        self.wait(0.5)
        self.play(FadeIn(check_spd, shift=RIGHT))
        self.wait(1.5)

        # Thu dọn màn hình để chuyển cảnh
        self.play(
            FadeOut(intro_text), 
            FadeOut(checks)
        )
        self.play(
            A_group.animate.to_corner(UL, buff=0.5).shift(DOWN).scale(0.8)
        )
        self.wait(0.5)

        # ==========================================
        # SCENE 2: PHÂN RÃ CHOLESKY
        # ==========================================
        cholesky_title = Text("1. Phân rã Cholesky", font_size=32, color=YELLOW).next_to(A_group, DOWN, buff=0.7).align_to(A_group, LEFT)
        self.play(Write(cholesky_title))

        formula = MathTex("A = L L^T").next_to(cholesky_title, RIGHT, buff=1)
        self.play(Write(formula))

        # Hiển thị khung ma trận L trống
        L_matrix = Matrix([["L_{00}", "0"], ["L_{10}", "L_{11}"]])
        L_label = MathTex("L = ")
        L_group = VGroup(L_label, L_matrix).arrange(RIGHT).shift(LEFT * 2)

        self.play(FadeIn(L_group, shift=UP))
        self.wait(0.5)

        # Trực quan hóa từng bước tính toán (Từ decomposition.py)
        calc_1 = MathTex("L_{00} = \\sqrt{4} = 2", font_size=36)
        calc_2 = MathTex("L_{10} = \\frac{2}{2} = 1", font_size=36)
        calc_3 = MathTex("L_{11} = \\sqrt{10 - 1^2} = 3", font_size=36)
        
        calcs = VGroup(calc_1, calc_2, calc_3).arrange(DOWN, aligned_edge=LEFT).next_to(L_group, RIGHT, buff=1.5)

        # Tính L00
        self.play(Write(calc_1))
        L_matrix_step1 = Matrix([["2", "0"], ["L_{10}", "L_{11}"]]).move_to(L_matrix)
        self.play(Transform(L_matrix, L_matrix_step1))
        self.wait(0.5)

        # Tính L10
        self.play(Write(calc_2))
        L_matrix_step2 = Matrix([["2", "0"], ["1", "L_{11}"]]).move_to(L_matrix)
        self.play(Transform(L_matrix, L_matrix_step2))
        self.wait(0.5)

        # Tính L11
        self.play(Write(calc_3))
        L_matrix_step3 = Matrix([["2", "0"], ["1", "3"]]).move_to(L_matrix)
        self.play(Transform(L_matrix, L_matrix_step3))
        self.wait(1)

        self.play(FadeOut(calcs))

        # Kiểm chứng: L * L^T
        verify_text = Text("Kiểm chứng:", font_size=28, color=BLUE).next_to(L_group, RIGHT, buff=1)
        
        L_final = Matrix([[2, 0], [1, 3]])
        LT_final = Matrix([[2, 1], [0, 3]])
        A_final = Matrix([[4, 2], [2, 10]])
        
        verify_eq = VGroup(
            L_final,
            MathTex("\\times"),
            LT_final,
            MathTex("="),
            A_final
        ).arrange(RIGHT).next_to(verify_text, DOWN, buff=0.5).align_to(verify_text, LEFT)

        self.play(FadeIn(verify_text, shift=DOWN))
        self.play(Write(verify_eq))
        self.wait(2)

        # Dọn dẹp không gian của Scene 2
        self.play(
            FadeOut(L_group), 
            FadeOut(verify_text), 
            FadeOut(verify_eq), 
            FadeOut(cholesky_title), 
            FadeOut(formula)
        )
        self.wait(0.5)

        # ==========================================
        # SCENE 3: CHÉO HÓA MA TRẬN
        # ==========================================
        diag_title = Text("2. Chéo hóa (Thuật toán QR Iteration)", font_size=32, color=YELLOW).next_to(A_group, DOWN, buff=0.7).align_to(A_group, LEFT)
        self.play(Write(diag_title))

        diag_formula = MathTex("A = P D P^{-1}").next_to(diag_title, RIGHT, buff=1)
        self.play(Write(diag_formula))

        # Trực quan hóa kết quả ma trận D và P (Lấy từ diagonalization.py: eigenvalue/vector)
        # Các vector riêng tương ứng lambda1 = 10.61 và lambda2 = 3.39
        D_matrix = Matrix([["10.61", "0"], ["0", "3.39"]])
        D_label = MathTex("D \\approx")
        D_group = VGroup(D_label, D_matrix).arrange(RIGHT)

        P_matrix = Matrix([["0.29", "-0.96"], ["0.96", "0.29"]])
        P_label = MathTex("P \\approx")
        P_group = VGroup(P_label, P_matrix).arrange(RIGHT)

        matrices_group = VGroup(D_group, P_group).arrange(RIGHT, buff=1.5).next_to(diag_title, DOWN, buff=1).align_to(diag_title, LEFT).shift(RIGHT * 1)

        self.play(FadeIn(D_group, shift=UP))
        self.wait(0.5)
        self.play(FadeIn(P_group, shift=UP))
        self.wait(1.5)

        # Ghép hoàn chỉnh phương trình A ≈ P * D * P^T
        A_reconstruct = MathTex("A \\approx")
        
        P_mat_small = Matrix([["0.29", "-0.96"], ["0.96", "0.29"]]).scale(0.8)
        D_mat_small = Matrix([["10.61", "0"], ["0", "3.39"]]).scale(0.8)
        PT_mat_small = Matrix([["0.29", "0.96"], ["-0.96", "0.29"]]).scale(0.8) # P^T
        
        reconstruct_eq = VGroup(
            A_reconstruct,
            P_mat_small,
            MathTex("\\times"),
            D_mat_small,
            MathTex("\\times"),
            PT_mat_small
        ).arrange(RIGHT).next_to(matrices_group, DOWN, buff=1).align_to(A_group, LEFT)

        self.play(Write(reconstruct_eq))
        self.wait(2)

        conclusion = Text("Hoàn thành chéo hóa!", font_size=36, color=GREEN).next_to(reconstruct_eq, DOWN, buff=1)
        self.play(FadeIn(conclusion, shift=UP))
        self.wait(3)

        # Kết thúc mượt mà
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)