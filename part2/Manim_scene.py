"""
Module: manim_scene.py
Description: Video trực quan hoá Phân rã Cholesky, QR (Gram‑Schmidt) và Chéo hoá ma trận cho Đồ án 1.

--- CÁCH CHẠY LỆNH TRÊN TERMINAL ---
Lưu ý: Nếu gặp lỗi thiếu thư viện LaTeX, hãy chạy lệnh này trước khi render:
export PATH="/usr/local/texlive/2025/bin/universal-darwin:$PATH"

1. Render tất cả các Scene (Chất lượng thấp 480p15 để test nhanh):
    manim -ql part2/Manim_scene.py IntroScene ProblemStatementScene Cholesky_scene QR_GramSchmidtScene EigenCalculationScene DiagonalizationScene GeometricInterpretation

2. Render chất lượng cao (1080p60 - tốn nhiều thời gian):
    manim -qh part2/Manim_scene.py IntroScene ProblemStatementScene Cholesky_scene QR_GramSchmidtScene EigenCalculationScene DiagonalizationScene GeometricInterpretation

3. Nối tất cả video lại thành 1 file duy nhất (Full_Project.mp4):
    ffmpeg -y -f concat -safe 0 -i media/videos/Manim_scene/480p15/videos.txt -c copy media/videos/Manim_scene/480p15/Full_Project.mp4
"""

from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Color theme (tùy chỉnh, vừa hiện đại vừa dễ nhìn)
# ---------------------------------------------------------------------------
COLOR_THEME = {
    "title": BLUE,
    "subtitle": YELLOW,
    "highlight": GREEN,
    "matrix": WHITE,
    "math": WHITE,
    "eigenvector": TEAL,
    "eigenvalue": ORANGE,
}

# ---------------------------------------------------------------------------
# 1. IntroScene – giới thiệu trường, môn học, tên đồ án và các thành viên
# ---------------------------------------------------------------------------
class IntroScene(Scene):
    def construct(self):
        Text.set_default(font="sans-serif")

        # Trường & môn học
        university = Text(
            "TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN",
            font_size=40,
            weight=BOLD,
        ).to_edge(UP, buff=1.0)
        subject = Text(
            "Môn học: Toán Ứng Dụng và Thống Kê",
            font_size=32,
            color=COLOR_THEME["subtitle"],
        ).next_to(university, DOWN, buff=0.5)
        self.play(FadeIn(university, shift=DOWN), FadeIn(subject, shift=DOWN))
        self.wait(1)

        # Tên đồ án – bay vào từ hai phía
        project_title = Text(
            "ĐỒ ÁN 1: Ma Trận và Cơ Sở của Tính Toán Khoa Học",
            font_size=36,
            color=COLOR_THEME["title"],
        )
        self.play(
            FadeOut(university),
            FadeOut(subject),
            FadeIn(project_title, shift=LEFT),
            FadeIn(project_title.copy(), shift=RIGHT),
        )
        self.wait(1)
        self.play(FadeOut(Group(*self.mobjects)))

        # Các thành viên
        members = VGroup(
            Text("Nhóm trưởng: Nguyễn Thành Bảo - 24120023", font_size=28, color=COLOR_THEME["highlight"]),
            Text("Thành viên: Bùi Thị Thùy Dương - 24120044", font_size=28),
            Text("Thành viên: Nguyễn Khánh Hoàng - 24120055", font_size=28),
            Text("Thành viên: Lê Nguyễn Thùy Linh - 24120085", font_size=28),
            Text("Thành viên: Mai Anh Phúc Minh - 24120094", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        self.play(Write(members, run_time=3))
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)))

# ---------------------------------------------------------------------------
# 2. ProblemStatementScene – nêu yêu cầu và ma trận A (3×3)
# ---------------------------------------------------------------------------
class ProblemStatementScene(Scene):
    def construct(self):
        Text.set_default(font="sans-serif")
        title = Text("Nội dung thực hiện", font_size=36, color=YELLOW).to_edge(UP)
        self.play(Write(title))

        reqs = VGroup(
            Text("1. Phân rã Cholesky", font_size=36),
            Text("2. Phân rã QR (Gram‑Schmidt)", font_size=36),
            Text("3. Chéo hoá ma trận", font_size=36),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(title, DOWN, buff=1)
        for r in reqs:
            self.play(FadeIn(r, shift=RIGHT))
            self.wait(0.3)
        self.wait(1)

        # Ma trận A
        self.play(FadeOut(reqs))
        intro = Text(
            "Xét ma trận đầu vào (3×3):",
            font_size=36,
            color=COLOR_THEME["subtitle"],
        ).next_to(title, DOWN, buff=1)
        self.play(FadeIn(intro))
        A = Matrix([[4, 2, 2], [2, 4, 2], [2, 2, 4]])
        A_label = MathTex(r"A = ")
        A_group = VGroup(A_label, A).arrange(RIGHT).next_to(intro, DOWN, buff=0.8)
        self.play(DrawBorderThenFill(A_group))
        self.wait(2)
        self.play(
            FadeOut(title),
            FadeOut(intro),
            A_group.animate.to_corner(UL, buff=0.5).scale(0.7),
        )

# ---------------------------------------------------------------------------
# 3. Cholesky_scene – kiểm tra SPD, tính L từng bước, kiểm chứng L·Lᵀ = A
# ---------------------------------------------------------------------------

class Cholesky_scene(ThreeDScene):
    def construct(self):
        # ==========================================
        # PHẦN 1: Phân tích Cholesky (2D) - Dẫn dắt từng bước
        # ==========================================
        
        # --- TIÊU ĐỀ MỞ ĐẦU ---
        main_title = Text("1. Phân rã Cholesky", font_size=48, color=BLUE_C)
        self.play(Write(main_title))
        self.wait(1.5)
        
        main_title_corner = Text("1. Phân rã Cholesky", font_size=28, color=BLUE_C).to_corner(UL)
        self.play(Transform(main_title, main_title_corner))
        
        # --- BƯỚC 1: ĐỊNH NGHĨA ---
        current_title = Text("Bước 1: Cholesky là gì?", font_size=36, color=YELLOW).to_edge(UP)
        self.play(Write(current_title))
        chol_def = MathTex("A", "=", "L", "L^T")
        chol_def.set_color_by_tex_to_color_map({"A": RED, "L": YELLOW, "L^T": YELLOW})
        chol_desc = Text("Phân tích ma trận thành tích của ma trận tam giác dưới (L) và chuyển vị của nó.", font_size=24).next_to(chol_def, DOWN, buff=0.5)
        
        def_group = VGroup(chol_def, chol_desc).move_to(ORIGIN)
        self.play(Write(def_group))
        self.wait(2)
        # Xóa tiêu đề góc trái và nội dung định nghĩa để nhường chỗ hoàn toàn cho Bước 2
        self.play(FadeOut(def_group), FadeOut(main_title))

        # --- BƯỚC 2: KIỂM TRA ĐIỀU KIỆN SPD ---
        step2_title = Text("Bước 2: Kiểm tra tính Xác định dương (SPD)", font_size=36, color=YELLOW).to_edge(UP)
        self.play(Transform(current_title, step2_title))

        # Khai báo ma trận A ban đầu ở giữa, sau đó thu nhỏ đưa lên góc
        a_vals = [[4, 2, 2], [2, 4, 2], [2, 2, 4]]
        A_mat = Matrix(a_vals, v_buff=1.5, h_buff=2.0).set_color(RED)
        A_label = MathTex("A = ").next_to(A_mat, LEFT)
        A_group = VGroup(A_label, A_mat).move_to(ORIGIN)

        self.play(Write(A_group))
        self.wait(1)
        # Thay vì đưa lên góc UL quá cao, ta đưa sát cạnh trái nhưng nằm ngay dưới tiêu đề Bước 2
        self.play(A_group.animate.scale(0.8).next_to(current_title, DOWN, buff=0.5).to_edge(LEFT, buff=1.0))

        # Viết các điều kiện kiểm tra
        cond_1_txt = Text("1. Đối xứng:", font_size=24, color=GREEN)
        cond_1_math = MathTex(r"A = A^T", color=GREEN).scale(0.8)
        cond_1_ok = Text("(Thỏa mãn)", font_size=24, color=GREEN)
        cond_1 = VGroup(cond_1_txt, cond_1_math, cond_1_ok).arrange(RIGHT, buff=0.2)
        cond_2 = Text("2. Các định thức con chính (Leading Principal Minors) > 0:", font_size=20, color=BLUE)
        
        m1 = MathTex(r"\Delta_1 = 4 > 0").scale(0.8)
        m2 = MathTex(r"\Delta_2 = \det \begin{bmatrix} 4 & 2 \\ 2 & 4 \end{bmatrix} = 12 > 0").scale(0.8)
        m3 = MathTex(r"\Delta_3 = \det(A) = 32 > 0").scale(0.8)
        
        minors_group = VGroup(m1, m2, m3).arrange(DOWN, aligned_edge=LEFT).next_to(cond_2, DOWN, aligned_edge=LEFT, buff=0.3)
        spd_group = VGroup(cond_1, cond_2, minors_group).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(A_group, RIGHT, buff=1)

        self.play(Write(cond_1))
        self.wait(0.5)
        self.play(Write(cond_2))
        self.play(Write(minors_group), run_time=2)
        self.wait(1)

        spd_conc = Text("=> A là ma trận Đối xứng Xác định dương (SPD), đủ điều kiện Cholesky.", font_size=24, color=YELLOW).to_edge(DOWN, buff=1.0)
        self.play(Write(spd_conc))
        self.wait(2)

        # Xoá điều kiện, giữ lại ma trận A
        self.play(FadeOut(spd_group), FadeOut(spd_conc))

        # --- BƯỚC 3: KHỞI TẠO MA TRẬN L ---
        step3_title = Text("Bước 3: Khởi tạo Ma trận L", font_size=36, color=YELLOW).to_edge(UP)
        self.play(Transform(current_title, step3_title))

        l_sym = [
            ["l_{11}", "0", "0"],
            ["l_{21}", "l_{22}", "0"],
            ["l_{31}", "l_{32}", "l_{33}"]
        ]
        L_mat = Matrix(l_sym, v_buff=1.5, h_buff=2.0).set_color(YELLOW)
        L_label = MathTex(r"\implies L = ")
        
        # Tính toán vị trí đích đến của phương trình A => L ở ngay giữa màn hình (thu nhỏ 0.7 để vừa màn hình)
        target_eq_group = VGroup(A_group.copy().scale(1/0.8), L_label, L_mat).arrange(RIGHT, buff=0.5).scale(0.7).move_to(ORIGIN)
        
        # Phục hồi kích thước và di chuyển A_group gốc đến vị trí chuẩn
        self.play(A_group.animate.scale(0.7/0.8).move_to(target_eq_group[0]))
        self.play(Write(L_label), Write(L_mat))
        self.wait(1)

        # Định nghĩa lại group thực tế đang ở trên màn hình
        eq_group = VGroup(A_group, L_label, L_mat)

        # --- BƯỚC 4: TÍNH TOÁN TỪNG BƯỚC L ---
        step4_title = Text("Bước 4: Tính toán từng bước L", font_size=36, color=YELLOW).to_edge(UP)
        self.play(Transform(current_title, step4_title))
        
        # Đẩy lên trên để nhường chỗ cho text tính toán (đã thu nhỏ từ Bước 3)
        self.play(eq_group.animate.next_to(current_title, DOWN, buff=0.3))

        steps = [
            (0, 0, r"l_{11} = \sqrt{a_{11}} = \sqrt{4} = 2", "2"),
            (1, 0, r"l_{21} = \frac{a_{21}}{l_{11}} = \frac{2}{2} = 1", "1"),
            (2, 0, r"l_{31} = \frac{a_{31}}{l_{11}} = \frac{2}{2} = 1", "1"),
            (1, 1, r"l_{22} = \sqrt{a_{22} - l_{21}^2} = \sqrt{4 - 1^2} = \sqrt{3}", r"\sqrt{3}"),
            (2, 1, r"l_{32} = \frac{a_{32} - l_{31}l_{21}}{l_{22}} = \frac{2 - (1)(1)}{\sqrt{3}} = \frac{1}{\sqrt{3}}", r"\frac{1}{\sqrt{3}}"),
            (2, 2, r"l_{33} = \sqrt{a_{33} - l_{31}^2 - l_{32}^2} = \sqrt{4 - 1 - \frac{1}{3}} = \sqrt{\frac{8}{3}}", r"\sqrt{\frac{8}{3}}")
        ]

        calc_text = MathTex("").to_edge(DOWN, buff=0.5)
        self.add(calc_text)

        # Xác định chính xác vị trí Mobject ma trận trong eq_group
        A_mat_ref = A_group[1] # A_group gồm [A_label, A_mat]
        L_mat_ref = L_mat      # L_mat đã là matrix

        for row, col, formula_str, result_str in steps:
            idx = row * 3 + col

            a_elem = A_mat_ref.get_entries()[idx]
            box_a = SurroundingRectangle(a_elem, color=RED)

            # Căn biểu thức tính toán sát biên dưới (bottom edge)
            new_calc = MathTex(formula_str).to_edge(DOWN, buff=0.5)
            self.play(Create(box_a), Transform(calc_text, new_calc), run_time=0.8)
            self.wait(1) 

            l_elem = L_mat_ref.get_entries()[idx]
            new_l_elem = MathTex(result_str).set_color(YELLOW).move_to(l_elem)
            box_l = SurroundingRectangle(l_elem, color=YELLOW)

            self.play(Create(box_l), run_time=0.5)
            self.play(Transform(l_elem, new_l_elem), run_time=0.8)
            self.wait(0.5)

            self.play(FadeOut(box_a), FadeOut(box_l), run_time=0.5)

        self.wait(1)
        
        # Dọn sạch sẽ ma trận cũ
        self.play(FadeOut(calc_text), FadeOut(eq_group))
        self.wait(0.5)

        # --- BƯỚC 5: KIỂM CHỨNG L * L^T = A ---
        step5_title = Text("Bước 5: Kiểm chứng L·Lᵀ = A", font_size=36, color=YELLOW).to_edge(UP)
        self.play(Transform(current_title, step5_title))

        L_final = Matrix([
            ["2", "0", "0"],
            ["1", r"\sqrt{3}", "0"],
            ["1", r"\frac{1}{\sqrt{3}}", r"\sqrt{\frac{8}{3}}"]
        ], v_buff=1.2, h_buff=1.5).set_color(YELLOW)

        LT_final = Matrix([
            ["2", "1", "1"],
            ["0", r"\sqrt{3}", r"\frac{1}{\sqrt{3}}"],
            ["0", "0", r"\sqrt{\frac{8}{3}}"]
        ], v_buff=1.2, h_buff=1.5).set_color(YELLOW)

        A_final = Matrix(a_vals, v_buff=1.2, h_buff=1.5).set_color(RED)

        verify_eq = VGroup(L_final, LT_final, MathTex("="), A_final).arrange(RIGHT, buff=0.3).scale(0.85).move_to(ORIGIN)

        self.play(Write(verify_eq))
        self.wait(1)
        
        # Gắn hộp bao quanh toàn bộ ma trận (Bounding Box)
        box_verify = SurroundingRectangle(verify_eq, color=BLUE, buff=0.3, corner_radius=0.2)
        self.play(Create(box_verify), run_time=1.5)
        self.wait(3)

        # Dọn dẹp màn hình 2D để chuẩn bị chuyển sang 3D
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)

        # ==========================================
        # PHẦN 2: Không gian 3D - Hình học của các vector hàng
        # ==========================================
        
        # 1. Cài đặt góc nhìn 3D và trục tọa độ ban đầu
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        axes = ThreeDAxes(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            z_range=[-1, 4, 1],
            x_length=6,
            y_length=6,
            z_length=6
        )
        self.play(Create(axes))
        self.wait(0.5)

        # Tiêu đề cố định trên màn hình
        title_3d = Text("Bước 6: Ý nghĩa hình học của L", font_size=36, color=YELLOW).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title_3d)
        self.play(Write(title_3d))

        # Các giá trị tính toán
        sqrt3 = np.sqrt(3)
        inv_sqrt3 = 1 / sqrt3
        sqrt8_3 = np.sqrt(8/3)

        # 2. Vẽ 2 vector v1, v2 (Tạo nên mặt phẳng XY)
        v1 = Arrow3D(start=ORIGIN, end=axes.c2p(2, 0, 0), color=RED)
        v1_label = MathTex(r"\vec{v}_1 = \begin{bmatrix} 2 \\ 0 \\ 0 \end{bmatrix}")
        v1_label.set_color(RED).scale(0.6).move_to(axes.c2p(2.5, -0.5, 0))
        
        v2 = Arrow3D(start=ORIGIN, end=axes.c2p(1, sqrt3, 0), color=GREEN)
        v2_label = MathTex(r"\vec{v}_2 = \begin{bmatrix} 1 \\ \sqrt{3} \\ 0 \end{bmatrix}")
        v2_label.set_color(GREEN).scale(0.6).move_to(axes.c2p(0.5, 2.5, 0))

        self.play(Create(v1), Write(v1_label), run_time=1)
        self.play(Create(v2), Write(v2_label), run_time=1)
        self.wait(1)

        # 3. Tạo hình bình hành (mặt phẳng) từ 2 vector v1 và v2
        p0 = axes.c2p(0, 0, 0)
        p1 = axes.c2p(2, 0, 0)
        p2 = axes.c2p(3, sqrt3, 0) # v1 + v2
        p3 = axes.c2p(1, sqrt3, 0)
        
        plane = Polygon(p0, p1, p2, p3, color=BLUE, fill_opacity=0.3, stroke_width=2)
        self.play(FadeIn(plane))
        self.wait(1)

        # Xoay camera nhẹ để thấy rõ mặt phẳng
        self.move_camera(phi=65 * DEGREES, theta=60 * DEGREES, run_time=2)

        # 4. Vẽ vector v3 (Hàng 3 của L) vươn ra khỏi mặt phẳng
        v3 = Arrow3D(start=ORIGIN, end=axes.c2p(1, inv_sqrt3, sqrt8_3), color=YELLOW)
        # Tối giản nhãn của v3, bỏ toạ độ ma trận
        v3_label = MathTex(r"\vec{v}_3")
        v3_label.set_color(YELLOW).scale(0.8).move_to(axes.c2p(1.2, inv_sqrt3 + 0.3, sqrt8_3 + 0.3))

        self.play(Create(v3), Write(v3_label), run_time=1.5)
        self.wait(1)

        # ĐIỀU CHỈNH GÓC QUAY CAMERA: 
        # Hạ thấp phi xuống 80 độ để thấy chiều cao rõ hơn, theta lướt qua một khoảng dài
        self.move_camera(phi=80 * DEGREES, theta=10 * DEGREES, run_time=3)
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(6)
        self.stop_ambient_camera_rotation()

        # 5. Fade out dọn dẹp cảnh cuối
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
# ---------------------------------------------------------------------------
# 4. QR_GramSchmidtScene – trực giao hoá 3 vector trong không gian 3D
# ---------------------------------------------------------------------------
class QR_GramSchmidtScene(ThreeDScene):
    def construct(self):
        Text.set_default(font="sans-serif")
        main_title = Text("2. Phân rã QR (Gram‑Schmidt)", font_size=48, color=BLUE_C)
        self.add_fixed_in_frame_mobjects(main_title)
        self.play(Write(main_title))
        self.wait(1.5)

        main_title_corner = Text("2. Phân rã QR (Gram‑Schmidt)", font_size=28, color=BLUE_C).to_corner(UL)
        self.add_fixed_in_frame_mobjects(main_title_corner)
        self.play(Transform(main_title, main_title_corner))

        # Axes 3D
        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-5, 5, 1],
            x_length=8,
            y_length=8,
            z_length=8,
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes))
        self.begin_ambient_camera_rotation(rate=0.08)

        # Vectors a1, a2, a3 (từ ma trận A)
        a1 = np.array([4, 2, 2])
        a2 = np.array([2, 4, 2])
        a3 = np.array([2, 2, 4])
        arrow_a1 = Arrow3D(start=ORIGIN, end=axes.coords_to_point(*a1), color=RED)
        arrow_a2 = Arrow3D(start=ORIGIN, end=axes.coords_to_point(*a2), color=GREEN)
        arrow_a3 = Arrow3D(start=ORIGIN, end=axes.coords_to_point(*a3), color=BLUE)
        label_a1 = MathTex(r"a_1", color=RED).to_corner(UR).shift(LEFT * 2)
        label_a2 = MathTex(r"a_2", color=GREEN).next_to(label_a1, DOWN)
        label_a3 = MathTex(r"a_3", color=BLUE).next_to(label_a2, DOWN)
        self.add_fixed_in_frame_mobjects(label_a1, label_a2, label_a3)
        self.play(Create(arrow_a1), Write(label_a1))
        self.play(Create(arrow_a2), Write(label_a2))
        self.play(Create(arrow_a3), Write(label_a3))
        self.wait(1)

        step_note = Text("Trực giao hoá Gram‑Schmidt", font_size=24).to_corner(DL)
        self.add_fixed_in_frame_mobjects(step_note)
        self.play(Write(step_note))

        # Bước 1: e1 = a1/||a1||
        norm_a1 = np.linalg.norm(a1)
        e1 = a1 / norm_a1
        e1_vis = e1 * 3
        arrow_e1 = Arrow3D(start=ORIGIN, end=axes.coords_to_point(*e1_vis), color=YELLOW)
        label_e1 = MathTex(r"e_1", color=YELLOW).next_to(label_a1, LEFT, buff=1)
        self.add_fixed_in_frame_mobjects(label_e1)
        step1 = Text("Bước 1: Chuẩn hoá a₁ → e₁", font_size=20).next_to(step_note, UP, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(step1)
        self.play(
            ReplacementTransform(arrow_a1, arrow_e1),
            Write(label_e1),
            Write(step1),
        )
        self.wait(1)

        # Bước 2: e2 = (a2 - proj_{e1} a2)/||…||
        proj_a2_e1 = np.dot(a2, e1) * e1
        u2 = a2 - proj_a2_e1
        norm_u2 = np.linalg.norm(u2)
        e2 = u2 / norm_u2
        e2_vis = e2 * 3
        arrow_e2 = Arrow3D(start=ORIGIN, end=axes.coords_to_point(*e2_vis), color=ORANGE)
        label_e2 = MathTex(r"e_2", color=ORANGE).next_to(label_a2, LEFT, buff=1)
        self.add_fixed_in_frame_mobjects(label_e2)
        step2 = Text("Bước 2: Loại bỏ thành phần e₁ → e₂", font_size=20).next_to(step1, UP, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(step2)
        self.play(
            ReplacementTransform(arrow_a2, arrow_e2),
            Write(label_e2),
            Write(step2),
        )
        self.wait(1)

        # Bước 3: e3 = (a3 - proj_{e1} a3 - proj_{e2} a3)/||…||
        proj_a3_e1 = np.dot(a3, e1) * e1
        proj_a3_e2 = np.dot(a3, e2) * e2
        u3 = a3 - proj_a3_e1 - proj_a3_e2
        norm_u3 = np.linalg.norm(u3)
        e3 = u3 / norm_u3
        e3_vis = e3 * 3
        arrow_e3 = Arrow3D(start=ORIGIN, end=axes.coords_to_point(*e3_vis), color=PINK)
        label_e3 = MathTex(r"e_3", color=PINK).next_to(label_a3, LEFT, buff=1)
        self.add_fixed_in_frame_mobjects(label_e3)
        step3 = Text("Bước 3: Loại bỏ e₁, e₂ → e₃", font_size=20).next_to(step2, UP, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(step3)
        self.play(
            ReplacementTransform(arrow_a3, arrow_e3),
            Write(label_e3),
            Write(step3),
        )
        self.wait(2)

        # Gom lại thành ma trận Q và trở về góc nhìn 2D
        self.stop_ambient_camera_rotation()
        self.play(
            FadeOut(axes),
            FadeOut(arrow_e1), FadeOut(arrow_e2), FadeOut(arrow_e3),
            FadeOut(step_note), FadeOut(step1), FadeOut(step2), FadeOut(step3),
            FadeOut(label_a1), FadeOut(label_a2), FadeOut(label_a3),
        )
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        basis_text = Text(
            "Hệ trực chuẩn (Orthonormal Basis)",
            font_size=32,
            color=COLOR_THEME["highlight"],
        )
        self.add_fixed_in_frame_mobjects(basis_text)
        self.play(basis_text.animate.to_edge(UP, buff=2))
        Q = Matrix([
            [f"{e1[0]:.2f}", f"{e2[0]:.2f}", f"{e3[0]:.2f}"],
            [f"{e1[1]:.2f}", f"{e2[1]:.2f}", f"{e3[1]:.2f}"],
            [f"{e1[2]:.2f}", f"{e2[2]:.2f}", f"{e3[2]:.2f}"],
        ]).scale(0.8)
        Q_label = MathTex(r"Q = ")
        Q_group = VGroup(Q_label, Q).arrange(RIGHT)
        self.add_fixed_in_frame_mobjects(Q_group)
        self.play(FadeIn(Q_group, shift=UP))
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)))

class EigenCalculationScene(Scene):
    def construct(self):
        Text.set_default(font="sans-serif")
        
        # 1. TIÊU ĐỀ CHÍNH CHÉO HOÁ
        main_title = Text("3. Chéo hoá ma trận", font_size=48, color=BLUE_C)
        self.play(FadeIn(main_title, run_time=1))
        self.wait(1.5)

        self.play(FadeOut(main_title, run_time=1))

        # --- BƯỚC 1 & 2: TÌM TRỊ RIÊNG VÀ VECTƠ RIÊNG ---
        title1 = Text("Bước 1 & 2: Kết quả Trị riêng và Vectơ riêng", font_size=36, color=YELLOW).to_edge(UP)
        self.play(Write(title1))

        # Hiển thị Ma trận A
        A_mat = Matrix([[4, 2, 2], [2, 4, 2], [2, 2, 4]]).scale(0.7)
        A_label = MathTex(r"A = ")
        A_group = VGroup(A_label, A_mat).arrange(RIGHT).next_to(title1, DOWN, buff=0.5)
        self.play(FadeIn(A_group, shift=UP))
        self.wait(1)

        # Kết quả Trị riêng và Vectơ riêng
        # Trường hợp lambda = 8
        case1_txt = Text("Với", font_size=28)
        case1_math = MathTex(r"\lambda_1 = 8:", font_size=40)
        case1_math[0][0:2].set_color(COLOR_THEME["eigenvalue"])
        case1 = VGroup(case1_txt, case1_math).arrange(RIGHT, buff=0.2).next_to(A_group, DOWN, buff=0.8).to_edge(LEFT, buff=1.5)
        
        v1_result = MathTex(r"\Rightarrow v_1 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}", font_size=36)
        v1_result.set_color(COLOR_THEME["eigenvector"])
        v1_result.next_to(case1, RIGHT, buff=0.8)
        
        self.play(Write(case1))
        self.play(FadeIn(v1_result, shift=LEFT, run_time=1.0))
        self.wait(1)

        # Trường hợp lambda = 2
        case2_txt = Text("Với", font_size=28)
        case2_math = MathTex(r"\lambda_2 = \lambda_3 = 2:", font_size=40)
        case2_math[0][0:2].set_color(COLOR_THEME["eigenvalue"])
        case2_math[0][3:5].set_color(COLOR_THEME["eigenvalue"])
        case2 = VGroup(case2_txt, case2_math).arrange(RIGHT, buff=0.2).next_to(case1, DOWN, buff=1.5).align_to(case1, LEFT)

        v23_result = MathTex(r"\Rightarrow v_2 = \begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix}, \quad v_3 = \begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix}", font_size=36)
        v23_result.set_color(COLOR_THEME["eigenvector"])
        v23_result.next_to(case2, RIGHT, buff=0.8)

        self.play(Write(case2))
        self.play(FadeIn(v23_result, shift=LEFT, run_time=1.0))
        self.wait(2.5)
        
        # Kết luận chuẩn hóa
        note = Text("Chuẩn hóa các vectơ này, ta sẽ ráp được ma trận P", font_size=28, color=COLOR_THEME["highlight"]).to_edge(DOWN, buff=0.5)
        self.play(Write(note))
        self.wait(3)

        self.play(FadeOut(Group(*self.mobjects)))
class DiagonalizationScene(Scene):
    def construct(self):
        Text.set_default(font="sans-serif")
        
        subtitle = Text("Bước 3: Lập công thức phân rã ma trận", font_size=36, color=YELLOW).to_edge(UP)
        self.play(FadeIn(subtitle))
        self.wait(1) # Dừng 1 giây để đọc tiêu đề

        def update_subtitle(text):
            new_sub = Text(text, font_size=36, color=YELLOW).to_edge(UP)
            self.play(Transform(subtitle, new_sub, run_time=1))

        # 2. HIỂN THỊ MA TRẬN A & CÔNG THỨC
        A = Matrix([[4, 2, 2], [2, 4, 2], [2, 2, 4]]).scale(0.7)
        A_label = MathTex(r"A = ")
        A_group = VGroup(A_label, A).arrange(RIGHT).center()
        self.play(FadeIn(A_group, run_time=1.5))
        self.wait(1)

        formula = MathTex(r"A = ", r"P", r"D", r"P^{-1}", font_size=48).next_to(A_group, DOWN, buff=0.8)
        formula.set_color_by_tex("P", COLOR_THEME["eigenvector"])
        formula.set_color_by_tex("D", COLOR_THEME["eigenvalue"])
        self.play(Write(formula, run_time=2)) # Viết công thức chậm hơn
        self.wait(2.5) # Dừng hẳn 2.5 giây để ghi nhớ công thức

        self.play(FadeOut(A_group), FadeOut(formula, run_time=1))
        self.wait(1)

        # 3. BƯỚC 4: XÂY DỰNG MA TRẬN D
        update_subtitle("Bước 4: Ma trận D (Chứa các Trị riêng trên đường chéo)")
        
        D = Matrix([["8", "0", "0"], ["0", "2", "0"], ["0", "0", "2"]]).scale(0.7)
        for i in [0, 4, 8]:
            D.get_entries()[i].set_color(COLOR_THEME["eigenvalue"])
            
        D_label = MathTex(r"D = ", color=COLOR_THEME["eigenvalue"])
        D_group = VGroup(D_label, D).arrange(RIGHT).move_to(LEFT * 3)

        self.play(FadeIn(D_label), Create(D.get_brackets(), run_time=1.5))
        self.play(Write(D.get_entries()), run_time=2.5) # Viết các con số chậm rãi
        self.wait(2) # Cho người xem thời gian nhìn đường chéo

        # 4. BƯỚC 5: XÂY DỰNG MA TRẬN P
        update_subtitle("Bước 5: Ma trận P (Ghép từ các Vectơ riêng)")
        
        P = Matrix([
            ["0.58", "0.71", "0.41"],
            ["0.58", "-0.71", "0.41"],
            ["0.58", "0", "-0.82"],
        ]).scale(0.7)
        for col in P.get_columns():
            col.set_color(COLOR_THEME["eigenvector"])

        P_label = MathTex(r"P \approx ", color=COLOR_THEME["eigenvector"])
        P_group = VGroup(P_label, P).arrange(RIGHT).next_to(D_group, RIGHT, buff=1)

        self.play(FadeIn(P_label), Create(P.get_brackets(), run_time=1.5))
        
        # Tăng lag_ratio và run_time để từng cột hiện ra rất rõ ràng
        self.play(
            Write(P.get_columns()[0]),
            Write(P.get_columns()[1]),
            Write(P.get_columns()[2]),
            run_time=3.5, lag_ratio=0.8 
        )
        self.wait(2.5)

        # 5. TÍNH CHẤT TRỰC GIAO CỦA MA TRẬN ĐỐI XỨNG
        update_subtitle("Lưu ý: Vì A là ma trận đối xứng nên P trực giao")
        
        note = MathTex(r"P^{-1} = P^T", font_size=36, color=YELLOW).next_to(subtitle, DOWN, buff=1.5)
        self.play(Write(note, run_time=1.5))
        self.wait(3) # Cần dừng lâu ở đây vì đây là kiến thức quan trọng

        # 6. BƯỚC 6: TÁI CẤU TRÚC LẠI A
        update_subtitle("Bước 6: Lắp ráp và phục hồi ma trận gốc A")

        PT = Matrix([
            ["0.58", "0.58", "0.58"],
            ["0.71", "-0.71", "0"],
            ["0.41", "0.41", "-0.82"],
        ]).scale(0.7)
        for row in PT.get_rows():
            row.set_color(COLOR_THEME["eigenvector"])

        expr = VGroup(
            MathTex(r"A \approx"),
            P.copy(),
            MathTex(r"\times"),
            D.copy(),
            MathTex(r"\times"),
            PT,
        ).arrange(RIGHT, buff=0.12).center().shift(DOWN * 0.5)

        self.play(FadeOut(D_group), FadeOut(P_group), FadeOut(note))
        self.play(FadeIn(expr), run_time=2)
        self.wait(2.5) # Để người xem nhìn lại cụm P * D * P^T

        A_final = Matrix([[4, 2, 2], [2, 4, 2], [2, 2, 4]]).scale(1.2)
        A_final_group = VGroup(MathTex(r"A ="), A_final).arrange(RIGHT).move_to(expr)
        
        self.play(ReplacementTransform(expr, A_final_group, run_time=3)) # Biến đổi mượt mà và chậm
        self.wait(2)

        complete = Text(
            "Hoàn tất phân rã & chéo hoá!",
            font_size=36,
            color=COLOR_THEME["highlight"],
            weight=BOLD,
        ).next_to(A_final_group, DOWN, buff=1)
        self.play(Write(complete, run_time=1.5))
        self.wait(4) # Dừng thật lâu ở màn hình kết quả cuối cùng
        
        self.play(FadeOut(Group(*self.mobjects), run_time=1.5))
class GeometricInterpretation(Scene):
    def construct(self):
        matrix = [[3, 1], [1, 3]]



        title = Text("Bước 7: Ý nghĩa hình học", font_size=36, color=YELLOW).to_edge(UP).add_background_rectangle()
        self.add(title)

        # Lưới tọa độ
        plane = NumberPlane()
        self.play(Create(plane), run_time=1.5)

        # Định nghĩa 2 vector riêng
        v1 = Vector([1, 1], color=ORANGE)
        v2 = Vector([-1, 1], color=ORANGE)

        # Label ban đầu
        v1_label = MathTex(r"v_1", color=ORANGE).next_to(v1.get_end(), RIGHT, buff=0.1).add_background_rectangle()
        v2_label = MathTex(r"v_2", color=ORANGE).next_to(v2.get_end(), UP, buff=0.1).add_background_rectangle()

        # Vẽ vector lên lưới
        self.play(Create(v1), Create(v2))
        self.play(FadeIn(v1_label), FadeIn(v2_label))
        self.wait(1)

        note = Text("Eigenvectors only get scaled, not rotated", font_size=24, color=YELLOW).to_edge(DOWN).add_background_rectangle()
        self.play(FadeIn(note))
        self.wait(1)

        # Xóa label và note để không bị dính vào hiệu ứng biến đổi
        self.play(FadeOut(v1_label), FadeOut(v2_label), FadeOut(note))

        # Thực thi phép biến đổi không gian bằng ApplyMatrix (biến đổi cả lưới và vector)
        self.play(
            ApplyMatrix(matrix, plane),
            ApplyMatrix(matrix, v1),
            ApplyMatrix(matrix, v2),
            run_time=3
        )
        self.wait(1)

        # Thêm label mới sau khi biến đổi (tỉ lệ với trị riêng lambda)
        v1_new_label = MathTex(r"\lambda_1 v_1", color=YELLOW).next_to(v1.get_end(), RIGHT, buff=0.1).add_background_rectangle()
        v2_new_label = MathTex(r"\lambda_2 v_2", color=YELLOW).next_to(v2.get_end(), UP, buff=0.1).add_background_rectangle()
        self.play(FadeIn(v1_new_label), FadeIn(v2_new_label))
        self.wait(3)