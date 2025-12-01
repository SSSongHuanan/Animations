from manim import *
import numpy as np
import random

class SARSADemo(Scene):
    def construct(self):
        # --- 0. 全局配置 ---
        self.gamma = 0.9
        self.alpha = 0.5
        self.grid_size = 3
        self.cell_size = 1.8
        self.grid_spacing = 2.0
        
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Q-Table
        self.q_table = np.zeros((self.grid_size, self.grid_size, 4))
        
        # --- 1. 理论对比 ---
        self.play_intro()
        
        # --- 2. 核心演示 ---
        self.play_sarsa_process()

    def play_intro(self):
        """对比 Q-Learning 和 SARSA 的公式"""
        title = Text("SARSA (On-Policy)", font_size=48, color=BLUE).to_edge(UP)
        self.play(Write(title))
        
        # Q-Learning 公式 (灰色，作为背景对比)
        q_learning_eq = MathTex(
            r"Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{a'} Q(S', a') - Q(S, A)]",
            font_size=32, color=GREY
        ).shift(UP * 1)
        q_label = Text("Q-Learning: Off-Policy (Greedy lookahead)", font_size=24, color=GREY).next_to(q_learning_eq, UP)
        
        self.play(FadeIn(q_label), Write(q_learning_eq))
        self.wait(1)
        
        # SARSA 公式 (高亮)
        sarsa_eq = MathTex(
            r"Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]",
            font_size=36, color=YELLOW
        ).shift(DOWN * 0.5)
        
        # 高亮 A'
        sarsa_eq[0][24:26].set_color(RED) # Q(S', A') 中的 A'
        
        s_label = Text("SARSA: On-Policy (Actual next action)", font_size=24, color=YELLOW).next_to(sarsa_eq, UP)
        
        explanation = VGroup(
            Text("Key Difference:", font_size=24),
            Text("SARSA uses the action it actually plans to take,", font_size=24, color=RED),
            Text("even if it's a bad exploration move.", font_size=24)
        ).arrange(DOWN).next_to(sarsa_eq, DOWN, buff=1)
        
        self.play(FadeIn(s_label), Write(sarsa_eq))
        self.play(Write(explanation))
        self.wait(3)
        
        self.play(
            FadeOut(q_learning_eq), FadeOut(q_label),
            FadeOut(explanation), FadeOut(s_label), FadeOut(sarsa_eq),
            title.animate.scale(0.8)
        )

    def get_q_color(self, value):
        """颜色映射"""
        if value > 0:
            return interpolate_color(GREY_E, GREEN, min(value, 1.0))
        elif value < 0:
            return interpolate_color(GREY_E, RED, min(abs(value), 1.0))
        else:
            return GREY_E

    def create_q_cell(self, r, c, pos, rewards):
        """创建三角形 Q-Cell"""
        half = self.cell_size / 2
        tl, tr = pos + [-half, half, 0], pos + [half, half, 0]
        bl, br = pos + [-half, -half, 0], pos + [half, -half, 0]
        center = pos
        
        if rewards[r, c] in [1.0, -1.0]:
            color = GREEN_E if rewards[r, c] == 1.0 else RED_E
            txt = "+1" if rewards[r, c] == 1.0 else "-1"
            sq = Square(side_length=self.cell_size, fill_color=color, fill_opacity=0.8, color=WHITE).move_to(pos)
            lbl = Text(txt, font_size=24).move_to(pos)
            return VGroup(sq, lbl), None
        
        # 4个三角形
        tris = []
        # Up, Down, Left, Right
        coords = [[center, tl, tr], [center, bl, br], [center, bl, tl], [center, br, tr]]
        for i, pts in enumerate(coords):
            tri = Polygon(*pts, color=WHITE, stroke_width=2, fill_color=BLACK, fill_opacity=0.6)
            tris.append(tri)
            
        label = None
        if rewards[r, c] == -0.5:
            label = Text("-0.5", color=ORANGE, font_size=20).move_to(pos).set_z_index(2)
            
        group = VGroup(*tris)
        if label: group.add(label)
        
        return group, tris

    def play_sarsa_process(self):
        # --- 环境初始化 ---
        rewards = np.array([
            [-0.04, -0.5, -0.04],
            [-0.04, -1.0,  -0.04],
            [-0.04, -0.04,  1.0]
        ])
        
        # 构建网格
        self.q_mobjects = {} 
        grid_group = VGroup()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = np.array([(j-1)*self.grid_spacing, (1-i)*self.grid_spacing, 0])
                grp, tris = self.create_q_cell(i, j, pos, rewards)
                grid_group.add(grp)
                if tris: self.q_mobjects[(i, j)] = tris
                
        grid_group.move_to(ORIGIN)
        self.play(Create(grid_group))
        
        # --- 演示阶段 ---
        info_text = Text("Step-by-Step SARSA", font_size=28, color=YELLOW).to_corner(UL)
        self.play(Write(info_text))
        
        # Agent
        agent = Dot(radius=0.15, color=BLUE_A).set_z_index(10)
        
        # 辅助函数：获取物理坐标
        def get_pos(r, c):
            return np.array([(c-1)*self.grid_spacing, (1-r)*self.grid_spacing, 0])
            
        agent.move_to(get_pos(0, 0))
        self.play(FadeIn(agent))
        
        # 变量初始化
        curr_r, curr_c = 0, 0
        # SARSA 需要先选好第一个动作
        curr_a_idx = 1 # 假设第一步选 Down (1)
        
        # 【修复】删除了这里导致报错的 arrow_end 计算代码
        # 因为进入循环后，第一件事就是画当前箭头，所以这里不需要预先计算
        
        steps_log = [
            # 当前(r,c), 当前a, 下一步(nr,nc), 下一步a (SARSA的关键), 奖励
            # Path: (0,0) -D-> (1,0) -D-> (2,0) -R-> (2,1) -R-> (2,2)
            
            # Step 1: (0,0) Down -> (1,0). Next Action: Down
            ((0,0), 1, (1,0), 1, -0.04),
            
            # Step 2: (1,0) Down -> (2,0). Next Action: Right
            ((1,0), 1, (2,0), 3, -0.04),
            
            # Step 3: (2,0) Right -> (2,1). Next Action: Right
            ((2,0), 3, (2,1), 3, -0.04),
            
            # Step 4: (2,1) Right -> (2,2) [Goal]
            ((2,1), 3, (2,2), None, 1.0)
        ]
        
        # 右侧面板
        panel = VGroup(
            Text("Current: S, A", font_size=24, color=BLUE),
            Text("Reward: R", font_size=24, color=WHITE),
            Text("Next: S', A'", font_size=24, color=RED), # SARSA 特色
            Text("Update Q(S,A)...", font_size=24, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT, buff=1)
        self.play(FadeIn(panel))
        
        # 幽灵智能体 (用于显示 Next Action)
        ghost = Dot(radius=0.1, color=RED).set_opacity(0.5)
        
        for step_idx, (r, c, nr, nc, reward, next_a_idx, real_r) in enumerate(self.augment_steps(steps_log)):
            
            # 1. 显示当前动作
            # 【关键修复】正确计算 3D 向量
            # Grid (dr, dc) -> Manim (dx, dy, 0) = (dc, -dr, 0)
            dr, dc = self.actions[curr_a_idx]
            action_vec = np.array([dc, -dr, 0]) * 0.6
            
            curr_arrow = Arrow(agent.get_center(), agent.get_center()+action_vec, color=BLUE, buff=0)
            self.play(Create(curr_arrow), run_time=0.3)
            
            # 2. 执行移动 (State -> Next State)
            self.play(
                agent.animate.move_to(get_pos(nr, nc)),
                FadeOut(curr_arrow),
                run_time=0.6
            )
            
            # 3. 显示奖励
            r_val_text = Text(f"R={reward}", font_size=24).next_to(panel[1], RIGHT)
            self.play(FadeIn(r_val_text), run_time=0.2)
            
            # 4. 【SARSA 核心】 选择并展示下一个动作 A'
            target_val = 0
            
            if next_a_idx is not None:
                # 移动幽灵到当前位置
                ghost.move_to(agent.get_center())
                self.add(ghost)
                
                # 幽灵执行下一个动作 (Show Lookahead)
                ndr, ndc = self.actions[next_a_idx]
                next_vec = np.array([ndc, -ndr, 0]) * 0.6
                
                ghost_arrow = Arrow(ghost.get_center(), ghost.get_center()+next_vec, color=RED, buff=0, stroke_width=3)
                
                self.play(
                    FadeIn(ghost_arrow),
                    Indicate(panel[2], color=RED), # 高亮 Next S', A' 文字
                    run_time=0.5
                )
                
                # 获取 Q(S', A')
                next_q = self.q_table[nr, nc, next_a_idx]
                target_val = reward + self.gamma * next_q
                
                # 高亮 Q-Table 中对应的三角形 (Next Action Triangle)
                if (nr, nc) in self.q_mobjects:
                    next_tri = self.q_mobjects[(nr, nc)][next_a_idx]
                    self.play(next_tri.animate.set_stroke(RED, width=4), run_time=0.3)
                    self.play(next_tri.animate.set_stroke(WHITE, width=2), run_time=0.2)
                
                self.play(FadeOut(ghost), FadeOut(ghost_arrow))
                
            else:
                # 终点
                target_val = reward
            
            # 5. 更新 Q(S, A)
            old_q = self.q_table[r, c, curr_a_idx]
            new_q = old_q + self.alpha * (target_val - old_q)
            self.q_table[r, c, curr_a_idx] = new_q
            
            # 可视化更新
            if (r, c) in self.q_mobjects:
                curr_tri = self.q_mobjects[(r, c)][curr_a_idx]
                new_color = self.get_q_color(new_q)
                
                self.play(Indicate(panel[3], color=YELLOW))
                self.play(
                    curr_tri.animate.set_fill(new_color, opacity=0.8),
                    run_time=0.5
                )
            
            self.play(FadeOut(r_val_text))
            
            # 准备下一轮
            curr_r, curr_c = nr, nc
            curr_a_idx = next_a_idx
            
            if next_a_idx is None: # Done
                self.play(Flash(agent, color=GREEN))
                self.play(FadeOut(agent))
                break
                
        # --- 快速收敛演示 ---
        self.play(
            FadeOut(panel),
            FadeOut(info_text),
            Write(Text("Converging...", font_size=36).to_corner(UL))
        )
        
        # 快速填充 Q 表颜色
        self.fill_final_q_table(rewards)
        
        # 画出最终安全路径
        path_arrows = VGroup()
        waypoints = [DOWN, DOWN, RIGHT, RIGHT]
        curr_pos = get_pos(0, 0)
        
        for vec in waypoints:
            arrow = Arrow(curr_pos, curr_pos + vec*0.6, color=GREEN, buff=0, stroke_width=5)
            path_arrows.add(arrow)
            curr_pos += vec * self.grid_spacing
            
        self.play(Create(path_arrows))
        
        final_text = Text("SARSA Policy Learned", color=GREEN).to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(2)

    def augment_steps(self, steps):
        """辅助函数：整理步骤数据"""
        # 将 raw list 转换为生成器，方便解压
        # 格式: (r, c, nr, nc, reward, next_a_idx, reward)
        # 上面的 list 少了一个参数，稍微修正下
        # Output: r, c, nr, nc, reward, next_a_idx, reward
        augmented = []
        for item in steps:
            # item: ((r,c), a, (nr,nc), na, rew)
            r, c = item[0]
            a = item[1]
            nr, nc = item[2]
            na = item[3]
            rew = item[4]
            augmented.append((r, c, nr, nc, rew, na, rew))
        return augmented

    def fill_final_q_table(self, rewards):
        """填充最终颜色"""
        anims = []
        # 简单模拟一些值
        for r in range(3):
            for c in range(3):
                if (r,c) not in self.q_mobjects: continue
                for i in range(4):
                    # 随机一点深浅不一的绿色或红色，假装学完了
                    val = np.random.uniform(-0.2, 0.2)
                    # 关键路径设为高分
                    if (r,c) == (0,0) and i==1: val = 0.5
                    if (r,c) == (1,0) and i==1: val = 0.6
                    if (r,c) == (2,0) and i==3: val = 0.7
                    if (r,c) == (2,1) and i==3: val = 0.9
                    
                    color = self.get_q_color(val)
                    anims.append(self.q_mobjects[(r,c)][i].animate.set_fill(color, opacity=0.8))
        self.play(*anims, run_time=1.5)