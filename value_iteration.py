from manim import *
import numpy as np

class ValueIterationFinal(Scene):
    def construct(self):
        # --- 0. 全局配置 ---
        self.gamma = 0.9
        self.grid_size = 3
        self.cell_size = 1.5
        self.grid_spacing = 1.6
        
        # --- 1. 播放理论介绍 ---
        self.play_intro()
        
        # --- 2. 播放核心网格演示 ---
        self.play_grid_world()

    def play_intro(self):
        """展示公式和参数的介绍场景"""
        self.title = Text("Value Iteration", font_size=48, color=BLUE).to_edge(UP)
        self.play(Write(self.title))
        
        bellman_eq = MathTex(
            r"V_{k+1}(s) = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')]",
            font_size=36
        ).shift(UP * 1)
        
        explanation = VGroup(
            Text("Core Idea:", font_size=24, color=YELLOW),
            Text("Iteratively propagate future rewards", font_size=24),
            Text("back to current state via discount factor γ", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(bellman_eq, DOWN, buff=0.5)
        
        # 【修改点 1】在介绍中加入 Mud (泥潭) 的参数说明
        params = VGroup(
            MathTex(r"\gamma (Discount\ Factor) = 0.9", color=ORANGE),
            # 这里增加了 Mud: -0.5
            Text("Goal: +1.0 | Trap: -1.0 | Mud: -0.5 | Step: -0.04", font_size=24, color=WHITE)
        ).arrange(DOWN).to_edge(DOWN, buff=1)

        self.play(Write(bellman_eq))
        self.play(FadeIn(explanation))
        self.play(Write(params))
        self.wait(3) # 多留一点时间给观众看新参数
        
        # 清理屏幕
        self.play(
            FadeOut(bellman_eq), 
            FadeOut(explanation), 
            FadeOut(params),
            self.title.animate.scale(0.8) 
        )

    def get_heatmap_color(self, value):
        """根据价值大小返回动态热力图颜色"""
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
        """主演示流程"""
        
        # --- A. 环境初始化 (打破对称性) ---
        # 【修改点 2】修改 rewards 矩阵，(0, 1) 设置为 -0.5
        rewards = np.array([
            [-0.04, -0.5, -0.04],   # 中间改为 -0.5 (泥潭)
            [-0.04, -1.0,  -0.04],  # 中间是 -1.0 (陷阱)
            [-0.04, -0.04,  1.0]    # 右下是 +1.0 (目标)
        ])
        
        values = np.zeros((self.grid_size, self.grid_size))
        values[2, 2] = 1.0 
        values[1, 1] = -1.0
        
        # --- B. 构建初始网格 ---
        grid_group = VGroup()
        self.value_trackers = {} 
        self.cells = {} 
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = (j - 1) * self.grid_spacing
                y = (1 - i) * self.grid_spacing
                pos = np.array([x, y, 0])
                
                # 初始颜色
                fill_color = self.get_heatmap_color(values[i, j])
                
                # 【修改点 3】特殊的格子给予特殊的初始颜色
                if rewards[i, j] == 1.0: 
                    fill_color = GREEN_E 
                elif rewards[i, j] == -1.0: 
                    fill_color = RED_E
                elif rewards[i, j] == -0.5: # 泥潭显示为橙色
                    fill_color = ORANGE

                cell = Square(side_length=self.cell_size, color=WHITE, fill_color=fill_color, fill_opacity=0.6)
                cell.move_to(pos)
                self.cells[(i, j)] = cell 
                
                # 奖励标签
                if rewards[i, j] == 1.0: r_text = "+1.0"
                elif rewards[i, j] == -1.0: r_text = "-1.0"
                elif rewards[i, j] == -0.5: r_text = "-0.5" # 显示泥潭数值
                else: r_text = "-0.04"
                
                reward_label = Text(r_text, font_size=16, color=YELLOW)
                reward_label.move_to(pos + np.array([-self.cell_size/2 + 0.35, self.cell_size/2 - 0.2, 0]))

                # 数值追踪器
                tracker = ValueTracker(values[i, j])
                self.value_trackers[(i, j)] = tracker
                
                val_num = DecimalNumber(
                    values[i, j], 
                    num_decimal_places=2, 
                    show_ellipsis=False,
                    font_size=30,
                    color=WHITE
                )
                val_num.move_to(pos)
                val_num.add_updater(lambda m, r=i, c=j: m.set_value(self.value_trackers[(r, c)].get_value()))
                
                grid_group.add(cell, reward_label, val_num)

        grid_group.move_to(ORIGIN)
        self.play(Create(grid_group))
        
        # --- C. 微观演示 (One-Step Lookahead) ---
        self.play(FadeOut(self.title)) 
        self.visualize_one_step(grid_group, values, rewards)
        
        # --- D. 布局转换 ---
        self.play(grid_group.animate.scale(0.8).to_edge(LEFT, buff=1))
        
        # 创建坐标轴
        ax = Axes(
            x_range=[0, 7, 1],       
            y_range=[0, 1.2, 0.5],   
            x_length=5,              
            y_length=4,
            axis_config={"color": BLUE},
            x_axis_config={"numbers_to_include": range(0, 8)},
            tips=False
        ).to_edge(RIGHT, buff=1)
        
        x_label = ax.get_x_axis_label("Iteration", edge=DOWN, direction=DOWN, buff=0.2).scale(0.6)
        y_label = ax.get_y_axis_label("Max Error", edge=LEFT, direction=LEFT, buff=0.2).scale(0.6).rotate(90*DEGREES)
        
        plot_group = VGroup(ax, x_label, y_label)
        self.play(Create(plot_group))
        
        iter_label = Text("Iteration: 0", font_size=24).next_to(grid_group, UP)
        self.play(Write(iter_label))
        
        # --- E. 宏观演示: 价值迭代主循环 ---
        max_iterations = 6 
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        goal_pos = (2, 2) 
        last_dot = None 

        for k in range(max_iterations):
            new_values = values.copy()
            anim_data = [] 
            
            self.play(Transform(iter_label, Text(f"Iteration: {k+1}", font_size=24).move_to(iter_label)), run_time=0.5)
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # 跳过固定格子 (现在包括泥潭吗？不，泥潭的值是会变化的，只有终点和陷阱通常视为吸收态不更新)
                    # 注意：通常 Value Iteration 中，如果陷阱是 Terminal State，则不更新。
                    # 泥潭 (-0.5) 只是 step cost 高，并不是终点，所以它应该参与更新！
                    # 这里只跳过 1.0 和 -1.0
                    if rewards[i, j] == 1.0 or rewards[i, j] == -1.0:
                        continue
                    
                    q_values = []
                    for di, dj in actions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            v_next = values[ni, nj]
                        else:
                            v_next = values[i, j]
                        
                        # Bellman Update
                        q = rewards[i, j] + self.gamma * v_next
                        q_values.append(q)
                    
                    new_values[i, j] = max(q_values)
                    
                    # 准备动画
                    dist = abs(i - goal_pos[0]) + abs(j - goal_pos[1])
                    val_anim = self.value_trackers[(i, j)].animate.set_value(new_values[i, j])
                    
                    # 热力图颜色更新 (泥潭格子也会随价值变化颜色，这没问题，或者你可以保持它为橙色)
                    # 如果希望泥潭保持橙色以示区别，可以加个判断。
                    # 但为了展示价值传播，让它变色效果更好。
                    # 不过为了视觉清晰，我们保留 -0.5 格子的特殊性可能更好？
                    # 权衡之下：让它参与热力图变化能体现出它“虽然是坑但离终点近”的价值属性。
                    target_color = self.get_heatmap_color(new_values[i, j])
                    
                    # 如果你非常希望泥潭一直显示橙色，可以取消下面这行对泥潭的颜色更新
                    # if rewards[i, j] != -0.5: 
                    color_anim = self.cells[(i, j)].animate.set_fill(target_color, opacity=0.6)
                    
                    group_anim = AnimationGroup(val_anim, color_anim)
                    anim_data.append((dist, group_anim))
            
            # 绘制图表
            diff_matrix = np.abs(new_values - values)
            max_delta = np.max(diff_matrix)
            
            current_point = ax.c2p(k + 1, max_delta)
            new_dot = Dot(current_point, color=YELLOW, radius=0.08)
            plot_anims = [FadeIn(new_dot, scale=0.5)]
            
            if last_dot is not None:
                line = Line(last_dot.get_center(), new_dot.get_center(), color=YELLOW, stroke_width=3)
                plot_anims.append(Create(line))
            
            # 播放动画
            values = new_values
            anim_data.sort(key=lambda x: x[0])
            sorted_grid_anims = [item[1] for item in anim_data]
            
            self.play(
                LaggedStart(*sorted_grid_anims, lag_ratio=0.15), 
                AnimationGroup(*plot_anims),                     
                run_time=1.5
            )
            
            last_dot = new_dot

        # 收敛标记
        converged_label = Text("Converged!", color=GREEN, font_size=36).next_to(last_dot, UP)
        self.play(Write(converged_label))
        self.wait(1)

        # --- F. 策略可视化 ---
        for mobj in grid_group:
            if isinstance(mobj, DecimalNumber):
                mobj.clear_updaters()

        self.play(grid_group.animate.set_opacity(0.3))
        
        arrows = VGroup()
        action_vectors = [UP, DOWN, LEFT, RIGHT]
        action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if rewards[i, j] == 1.0 or rewards[i, j] == -1.0:
                    continue
                
                best_val = -np.inf
                best_vec = UP
                pos = self.cells[(i, j)].get_center()
                
                for vec, (di, dj) in zip(action_vectors, action_deltas):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                        val = values[ni, nj]
                    else:
                        val = values[i, j]
                    
                    if val > best_val:
                        best_val = val
                        best_vec = vec
                
                arrow = Arrow(
                    start=ORIGIN, end=best_vec * 0.4, 
                    buff=0, color=GOLD, stroke_width=6,
                    max_tip_length_to_length_ratio=0.4
                ).move_to(pos)
                arrows.add(arrow)
        
        self.play(Create(arrows), run_time=2)
        self.wait(0.5)

        # --- G. 智能体实战 ---
        self.simulate_agent(values, start_pos=(0, 0))
        
        final_text = Text("Optimal Path Found", color=GREEN, font_size=36).to_edge(DOWN)
        self.play(FadeIn(final_text))
        self.wait(3)

    def visualize_one_step(self, grid_group, values, rewards):
        """可视化单步前瞻"""
        r, c = 1, 2 
        target_cell = self.cells[(r, c)]
        
        self.play(grid_group.animate.set_opacity(0.15))
        
        highlight_box = Square(side_length=self.cell_size, color=YELLOW, stroke_width=8, fill_opacity=0)
        highlight_box.move_to(target_cell.get_center())
        
        title_text = Text("One-Step Lookahead", color=YELLOW, font_size=36).to_edge(UP)
        
        self.play(
            Create(highlight_box),
            target_cell.animate.set_opacity(1),
            Write(title_text)
        )
        
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        vecs = [UP, DOWN, LEFT, RIGHT]
        
        arrows = VGroup()
        calc_labels = VGroup()
        q_vals = []
        
        for idx, (di, dj) in enumerate(actions):
            ni, nj = r + di, c + dj
            start = target_cell.get_center()
            direction_vec = vecs[idx]
            end = start + direction_vec * (self.grid_spacing * 0.7)
            
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                v_neighbor = values[ni, nj]
            else:
                v_neighbor = values[r, c] 
            
            current_reward = rewards[r, c]
            q_val = current_reward + self.gamma * v_neighbor
            q_vals.append(q_val)
            
            arrow = Arrow(start, end, buff=0, color=BLUE_B, stroke_width=4, max_tip_length_to_length_ratio=0.25)
            arrows.add(arrow)
            
            label_text = MathTex(
                f"{current_reward} + 0.9({v_neighbor:.1f})", 
                font_size=20, 
                color=WHITE
            )
            
            if v_neighbor == 1.0: label_text.set_color(GREEN)
            elif v_neighbor == -1.0: label_text.set_color(RED)
            elif v_neighbor < 0: label_text.set_color(ORANGE) # 稍微提示一下负值
            
            text_bg = BackgroundRectangle(label_text, color=BLACK, fill_opacity=0.7, buff=0.05)
            label_group = VGroup(text_bg, label_text)
            label_group.next_to(end, direction_vec, buff=0.1)
            
            calc_labels.add(label_group)
        
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2))
        self.play(FadeIn(calc_labels))
        self.wait(0.5)
        
        result_objs = VGroup()
        for i, q in enumerate(q_vals):
            res_text = MathTex(f"= {q:.2f}", font_size=24, color=YELLOW)
            target_pos = calc_labels[i].get_center() + DOWN * 0.35
            res_text.move_to(target_pos)
            res_bg = BackgroundRectangle(res_text, color=BLACK, fill_opacity=0.8, buff=0.05)
            group = VGroup(res_bg, res_text)
            result_objs.add(group)
            
        self.play(FadeIn(result_objs))
        self.wait(1)
        
        max_q = max(q_vals)
        max_idx = q_vals.index(max_q)
        
        best_group = result_objs[max_idx]
        best_bg = best_group[0]
        best_text = best_group[1]
        
        self.play(
            best_text.animate.scale(1.5).set_color(GREEN),
            *[FadeOut(result_objs[i]) for i in range(4) if i != max_idx],
            FadeOut(best_bg), 
            FadeOut(calc_labels),
            FadeOut(arrows)
        )
        
        final_text = Text("MAX", font_size=24, color=GREEN).next_to(target_cell, UP, buff=0.1)
        self.play(Write(final_text))
        
        self.play(
            best_text.animate.move_to(target_cell.get_center()),
            run_time=0.8
        )
        
        self.value_trackers[(r, c)].set_value(max_q)
        
        self.play(
            FadeOut(best_text), 
            FadeOut(final_text), 
            FadeOut(highlight_box), 
            FadeOut(title_text)
        )
        
        self.play(grid_group.animate.set_opacity(1))
        self.wait(1)

    def simulate_agent(self, values, start_pos=(0, 0)):
        """模拟智能体实战"""
        agent = Dot(color=BLUE_A, radius=0.2).set_z_index(10)
        start_cell = self.cells[start_pos]
        agent.move_to(start_cell.get_center())
        
        sim_text = Text("Agent Simulation", font_size=24, color=BLUE).next_to(self.cells[(0,0)], UP, buff=0.5)
        
        self.play(
            FadeIn(agent, scale=0.5),
            Write(sim_text)
        )
        
        path = TracedPath(agent.get_center, stroke_color=BLUE, stroke_opacity=0.6, stroke_width=5)
        self.add(path)
        
        curr_r, curr_c = start_pos
        steps = 0
        max_steps = 10 
        
        while steps < max_steps:
            best_val = -np.inf
            best_move = None
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
            
            for dr, dc in moves:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    val = values[nr, nc]
                    if val > best_val:
                        best_val = val
                        best_move = (nr, nc)
            
            if best_move:
                next_r, next_c = best_move
                target_cell = self.cells[(next_r, next_c)]
                
                self.play(agent.animate.move_to(target_cell.get_center()), run_time=0.6)
                curr_r, curr_c = next_r, next_c
                
                if values[curr_r, curr_c] == 1.0:
                    victory_text = Text("Goal Reached!", color=YELLOW, font_size=36).move_to(sim_text)
                    self.play(
                        Transform(sim_text, victory_text),
                        Flash(agent, color=YELLOW, flash_radius=0.5)
                    )
                    break
                elif values[curr_r, curr_c] == -1.0:
                    fail_text = Text("Trapped!", color=RED, font_size=36).move_to(sim_text)
                    self.play(Transform(sim_text, fail_text), agent.animate.set_color(RED))
                    break
            else:
                break
            steps += 1