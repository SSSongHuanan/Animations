from manim import *
import numpy as np
import random

class PolicyIterationDemo(Scene):
    def construct(self):
        # --- 0. 全局配置 ---
        self.gamma = 0.9
        self.grid_size = 3
        self.cell_size = 1.5
        self.grid_spacing = 1.6
        
        # 动作定义: 0:上, 1:下, 2:左, 3:右
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        self.action_vecs = [UP, DOWN, LEFT, RIGHT]
        
        # --- 1. 播放理论介绍 ---
        self.play_intro()
        
        # --- 2. 播放核心演示 ---
        self.play_grid_world()

    def play_intro(self):
        """展示策略迭代的核心逻辑"""
        title = Text("Policy Iteration", font_size=48, color=BLUE).to_edge(UP)
        self.play(Write(title))
        
        # 两个核心步骤的公式/文字
        step1 = VGroup(
            Text("1. Policy Evaluation", font_size=32, color=YELLOW),
            MathTex(r"V(s) \leftarrow \sum P(s'|s, \pi(s))[R + \gamma V(s')]", font_size=28)
        ).arrange(DOWN)
        
        step2 = VGroup(
            Text("2. Policy Improvement", font_size=32, color=GREEN),
            MathTex(r"\pi(s) \leftarrow \arg\max_a \sum P(s'|s, a)[R + \gamma V(s')]", font_size=28)
        ).arrange(DOWN)
        
        group = VGroup(step1, step2).arrange(RIGHT, buff=1.0).shift(UP*0.5)
        
        loop_arrow = CurvedArrow(step2.get_bottom(), step1.get_bottom(), angle=PI/2, color=WHITE)
        loop_text = Text("Repeat until stable", font_size=24).next_to(loop_arrow, DOWN)
        
        self.play(FadeIn(step1))
        self.wait(1)
        self.play(FadeIn(step2))
        self.wait(1)
        self.play(Create(loop_arrow), Write(loop_text))
        self.wait(2)
        
        self.play(
            FadeOut(step1), FadeOut(step2), FadeOut(loop_arrow), FadeOut(loop_text),
            title.animate.scale(0.8)
        )

    def get_heatmap_color(self, value):
        """热力图配色方案"""
        COLOR_NEUTRAL = BLACK     
        COLOR_POS = GREEN         
        COLOR_NEG = RED           
        
        if value > 0:
            alpha = min(value, 1.0) 
            return interpolate_color(COLOR_NEUTRAL, COLOR_POS, alpha)
        elif value < 0:
            alpha = min(abs(value), 1.0)
            return interpolate_color(COLOR_NEUTRAL, COLOR_NEG, alpha)
        else:
            return COLOR_NEUTRAL

    def play_grid_world(self):
        # --- A. 环境初始化 (带泥潭) ---
        rewards = np.array([
            [-0.04, -0.5, -0.04],   # (0,1) 是泥潭
            [-0.04, -1.0,  -0.04],  # (1,1) 是陷阱
            [-0.04, -0.04,  1.0]    # (2,2) 是目标
        ])
        
        values = np.zeros((self.grid_size, self.grid_size))
        values[2, 2] = 1.0 
        values[1, 1] = -1.0
        
        # 随机初始化策略 (每个格子随机选一个方向)
        # policy[i,j] 存储的是动作的索引 0-3
        np.random.seed(42) # 固定随机种子以便复现
        policy = np.random.randint(0, 4, size=(self.grid_size, self.grid_size))
        
        # --- B. 构建网格图形 ---
        grid_group = VGroup()
        self.value_trackers = {} 
        self.cells = {} 
        self.arrow_mobjects = {} # 存储策略箭头
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = np.array([(j - 1) * self.grid_spacing, (1 - i) * self.grid_spacing, 0])
                
                # 初始颜色
                fill_color = self.get_heatmap_color(values[i, j])
                if rewards[i, j] == 1.0: fill_color = GREEN_E 
                elif rewards[i, j] == -1.0: fill_color = RED_E
                elif rewards[i, j] == -0.5: fill_color = ORANGE

                cell = Square(side_length=self.cell_size, color=WHITE, fill_color=fill_color, fill_opacity=0.6)
                cell.move_to(pos)
                self.cells[(i, j)] = cell 
                
                # 标签
                if rewards[i, j] == 1.0: r_text = "+1.0"
                elif rewards[i, j] == -1.0: r_text = "-1.0"
                elif rewards[i, j] == -0.5: r_text = "-0.5"
                else: r_text = "-0.04"
                
                reward_label = Text(r_text, font_size=16, color=YELLOW).move_to(pos + np.array([-0.4, 0.55, 0]))
                
                # 数值
                tracker = ValueTracker(values[i, j])
                self.value_trackers[(i, j)] = tracker
                val_num = DecimalNumber(values[i, j], num_decimal_places=2, font_size=24, color=WHITE)
                val_num.move_to(pos + DOWN * 0.3) # 稍微往下移，给箭头腾位置
                val_num.add_updater(lambda m, r=i, c=j: m.set_value(self.value_trackers[(r, c)].get_value()))
                
                grid_group.add(cell, reward_label, val_num)
                
                # --- 初始化随机箭头 ---
                if not (rewards[i, j] == 1.0 or rewards[i, j] == -1.0):
                    action_idx = policy[i, j]
                    vec = self.action_vecs[action_idx]
                    arrow = Arrow(
                        start=ORIGIN, end=vec * 0.5, 
                        buff=0, color=GOLD, stroke_width=4,
                        max_tip_length_to_length_ratio=0.4
                    ).move_to(pos)
                    self.arrow_mobjects[(i, j)] = arrow
                    grid_group.add(arrow)

        # 布局
        grid_group.move_to(ORIGIN).scale(0.8).to_edge(LEFT, buff=1)
        self.play(Create(grid_group))
        
        # 状态指示器
        phase_text = Text("Initializing...", font_size=36).to_edge(RIGHT, buff=2)
        self.play(Write(phase_text))
        
        # --- C. 策略迭代主循环 ---
        iteration_count = 0
        policy_stable = False
        
        while not policy_stable:
            iteration_count += 1
            iter_indicator = Text(f"Iteration: {iteration_count}", font_size=24, color=GREY).next_to(phase_text, UP)
            self.add(iter_indicator)
            
            # --- 阶段 1: 策略评估 (Policy Evaluation) ---
            self.play(Transform(phase_text, Text("Policy Evaluation", color=YELLOW, font_size=36).move_to(phase_text)))
            
            # 视觉效果：淡化箭头，高亮数字
            self.play(
                *[a.animate.set_opacity(0.3) for a in self.arrow_mobjects.values()],
                run_time=0.5
            )
            
            # 运行多步 PE 直到数值相对稳定 (这里为了演示效果，固定运行 5 步)
            for pe_step in range(5): 
                new_values = values.copy()
                grid_anims = []
                
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if rewards[i, j] == 1.0 or rewards[i, j] == -1.0: continue
                        
                        # 关键：只根据当前策略 policy[i,j] 更新，不取 max
                        action_idx = policy[i, j]
                        di, dj = self.actions[action_idx]
                        ni, nj = i + di, j + dj
                        
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            v_next = values[ni, nj]
                        else:
                            v_next = values[i, j] # 撞墙
                            
                        # Bellman Expectation Equation
                        new_values[i, j] = rewards[i, j] + self.gamma * v_next
                        
                        # 动画
                        val_anim = self.value_trackers[(i, j)].animate.set_value(new_values[i, j])
                        # 颜色
                        new_color = self.get_heatmap_color(new_values[i, j])
                        if rewards[i, j] == -0.5: new_color = ORANGE # 保持泥潭颜色
                        col_anim = self.cells[(i, j)].animate.set_fill(new_color, opacity=0.6)
                        
                        grid_anims.append(AnimationGroup(val_anim, col_anim))
                
                values = new_values
                self.play(LaggedStart(*grid_anims, lag_ratio=0.05), run_time=0.5) # 快速播放 PE
            
            # --- 阶段 2: 策略提升 (Policy Improvement) ---
            self.play(Transform(phase_text, Text("Policy Improvement", color=GREEN, font_size=36).move_to(phase_text)))
            
            # 视觉效果：恢复箭头，淡化数字背景（可选）
            self.play(
                *[a.animate.set_opacity(1.0) for a in self.arrow_mobjects.values()],
                run_time=0.5
            )
            
            policy_stable = True
            improvement_anims = []
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if rewards[i, j] == 1.0 or rewards[i, j] == -1.0: continue
                    
                    old_action = policy[i, j]
                    
                    # 寻找贪婪动作
                    best_val = -np.inf
                    best_action = old_action
                    
                    for idx, (di, dj) in enumerate(self.actions):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            v_neighbor = values[ni, nj]
                        else:
                            v_neighbor = values[i, j]
                        
                        q_val = rewards[i, j] + self.gamma * v_neighbor
                        if q_val > best_val + 1e-5: # 加微小阈值防浮点误差
                            best_val = q_val
                            best_action = idx
                    
                    # 如果动作改变了，创建旋转动画
                    if best_action != old_action:
                        policy_stable = False
                        policy[i, j] = best_action
                        
                        arrow = self.arrow_mobjects[(i, j)]
                        new_vec = self.action_vecs[best_action]
                        
                        # 旋转动画
                        anim = Rotate(
                            arrow, 
                            angle=angle_between_vectors(arrow.get_vector(), new_vec),
                            about_point=arrow.get_start() # 绕起点旋转
                        )
                        # 或者直接 Transform
                        target_arrow = Arrow(start=ORIGIN, end=new_vec*0.5, buff=0, color=GREEN, stroke_width=4, max_tip_length_to_length_ratio=0.4).move_to(arrow.get_center())
                        anim = Transform(arrow, target_arrow)
                        
                        # 同时稍微高亮一下这个格子，表示策略变了
                        flash = self.cells[(i, j)].animate.set_stroke(YELLOW, width=6).set_stroke(WHITE, width=0) # 闪一下
                        improvement_anims.append(Succession(anim, flash))
            
            if len(improvement_anims) > 0:
                self.play(LaggedStart(*improvement_anims, lag_ratio=0.1), run_time=1.5)
            else:
                self.wait(0.5) # 如果没变化，稍作停留
            
            self.remove(iter_indicator)
            
            # 防止无限循环演示 (虽然 PI 理论上必收敛)
            if iteration_count > 10: break

        # --- D. 收敛完成 ---
        final_text = Text("Policy Converged!", color=GOLD, font_size=40).move_to(phase_text)
        self.play(Transform(phase_text, final_text))
        self.wait(1)

        # --- E. 智能体实战 (和之前一样) ---
        self.simulate_agent(values, start_pos=(0, 0))

    def simulate_agent(self, values, start_pos=(0, 0)):
        # 简化版模拟代码
        agent = Dot(color=BLUE_A, radius=0.2).set_z_index(10)
        start_cell = self.cells[start_pos]
        agent.move_to(start_cell.get_center())
        
        sim_text = Text("Agent Run", font_size=24, color=BLUE).next_to(self.cells[(0,0)], UP, buff=0.5)
        self.play(FadeIn(agent), Write(sim_text))
        path = TracedPath(agent.get_center, stroke_color=BLUE, stroke_opacity=0.6, stroke_width=5)
        self.add(path)
        
        curr_r, curr_c = start_pos
        steps = 0
        while steps < 10:
            # 简单的贪心策略
            best_val = -np.inf
            best_move = None
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if values[nr, nc] > best_val:
                        best_val = values[nr, nc]
                        best_move = (nr, nc)
            
            if best_move:
                next_r, next_c = best_move
                self.play(agent.animate.move_to(self.cells[(next_r, next_c)].get_center()), run_time=0.5)
                curr_r, curr_c = next_r, next_c
                if values[curr_r, curr_c] == 1.0:
                    self.play(Flash(agent, color=YELLOW))
                    break
            steps += 1
        self.wait(2)

def angle_between_vectors(v1, v2):
    """辅助函数：计算旋转角度"""
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    return angle2 - angle1