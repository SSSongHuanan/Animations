from manim import *
import numpy as np

class PolicyIterationGeneral(Scene):
    def construct(self):
        # --- 0. 全局配置 (5x5) ---
        config.max_files_cached = 500 
        self.gamma = 0.9        
        
        # 固定轮次配置: 第一轮 10 步，后续 5 步
        # 增加总轮次，确保策略能传导到起点
        self.pe_steps_schedule = [10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5] 
        
        self.grid_size = 5      
        self.cell_size = 1.0    
        self.grid_spacing = 1.1 
        
        # 动作定义
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        self.action_vecs = [UP, DOWN, LEFT, RIGHT]
        
        # --- 1. 播放理论介绍 ---
        self.play_intro()
        
        # --- 2. 播放核心网格演示 ---
        self.play_grid_world()

    def play_intro(self):
        """展示策略迭代 Intro"""
        # 1. 标题
        self.title = Text("Policy Iteration", font_size=48, color=BLUE).to_edge(UP)
        env_text = Text("Problem: 5x5 Grid Maze Navigation", font_size=32, color=TEAL).next_to(self.title, DOWN, buff=0.3)
        
        # 2. 内容 (左对齐)
        s1_title = Text("1. Policy Evaluation", font_size=28, color=YELLOW)
        s1_desc = Text("Iterate V(s) for k times (Fixed Steps)", font_size=20, color=GREY_B).next_to(s1_title, RIGHT, buff=0.2, aligned_edge=DOWN)
        s1_header = VGroup(s1_title, s1_desc)
        s1_eq = MathTex(r"V(s) \leftarrow \sum P(s'|s, \pi(s))[R + \gamma V(s')]", font_size=30).shift(RIGHT * 0.5)
        step1_group = VGroup(s1_header, s1_eq).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        s2_title = Text("2. Policy Improvement", font_size=28, color=GREEN)
        s2_eq = MathTex(r"\pi(s) \leftarrow \arg\max_a \sum P(s'|s, a)[R + \gamma V(s')]", font_size=30).shift(RIGHT * 0.5)
        step2_group = VGroup(s2_title, s2_eq).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        content_group = VGroup(step1_group, step2_group).arrange(DOWN, aligned_edge=LEFT, buff=0.8)
        content_group.next_to(env_text, DOWN, buff=0.8)
        
        # 3. 底部参数
        params_content = VGroup(
            MathTex(r"\gamma = 0.9", color=YELLOW),
            Text("k = 10 (Initial), k = 5 (Later)", font_size=24, color=RED),
            Text("| Goal:+1.0  Trap:-1.0  Mud:-0.5  Step:-0.04", font_size=20, color=GREY_B)
        ).arrange(RIGHT, buff=0.3)
        params = params_content.to_edge(DOWN, buff=1)

        # 4. 动画
        self.play(FadeIn(self.title), FadeIn(env_text))
        self.wait(0.5)
        self.play(FadeIn(step1_group, shift=RIGHT))
        self.wait(0.5)
        self.play(FadeIn(step2_group, shift=RIGHT))
        
        arrow_start = step2_group.get_left() + LEFT * 0.2
        arrow_end = step1_group.get_left() + LEFT * 0.2
        loop_arrow = CurvedArrow(arrow_start, arrow_end, angle=-PI/2, color=WHITE)
        loop_text = Text("Repeat", font_size=16).next_to(loop_arrow, LEFT)
        
        self.play(Create(loop_arrow), Write(loop_text))
        self.play(FadeIn(params))
        self.wait(2)
        
        self.play(
            FadeOut(content_group), 
            FadeOut(params),
            FadeOut(env_text),
            FadeOut(loop_arrow),
            FadeOut(loop_text),
            self.title.animate.scale(0.8) 
        )

    def get_static_color(self, reward):
        if reward == 1.0: return TEAL_E      
        if reward == -1.0: return MAROON_E   
        if reward == -0.5: return ORANGE     
        return DARK_GRAY                     

    def play_grid_world(self):
        # --- A. 环境初始化 ---
        rewards = np.array([
            [-0.04, -0.04, -0.04, -0.50, -1.00], 
            [-0.04, -1.00, -0.04, -1.00, -1.00], 
            [-0.04, -0.50, -0.04, -0.04, -0.04], 
            [-1.00, -1.00, -1.00, -1.00, -0.04], 
            [-1.00, -0.04, -0.04, -0.04,  1.00]  
        ])
        
        values = np.zeros((self.grid_size, self.grid_size))
        
        np.random.seed(42)
        policy = np.random.randint(0, 4, size=(self.grid_size, self.grid_size))
        
        # --- B. 构建网格 ---
        grid_group = VGroup()
        self.value_trackers = {} 
        self.cells = {} 
        self.arrows = {} 
        
        center_offset = (self.grid_size - 1) * self.grid_spacing / 2

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j * self.grid_spacing - center_offset
                y = (self.grid_size - 1 - i) * self.grid_spacing - center_offset
                pos = np.array([x, y, 0])
                
                fill_color = self.get_static_color(rewards[i, j])
                cell = Square(side_length=self.cell_size, color=WHITE, fill_color=fill_color, fill_opacity=0.8)
                cell.move_to(pos)
                self.cells[(i, j)] = cell 
                
                if rewards[i, j] == 1.0: r_text, label_color = "Goal\n+1.0", YELLOW
                elif rewards[i, j] == -1.0: r_text, label_color = "Trap\n-1.0", WHITE
                elif rewards[i, j] == -0.5: r_text, label_color = "Mud\n-0.5", WHITE
                else: r_text, label_color = "-0.04", WHITE
                
                reward_label = Text(r_text, font_size=9, line_spacing=0.8, color=label_color)
                reward_label.move_to(pos + np.array([0, 0.32, 0]))
                reward_label.set_z_index(10)

                tracker = ValueTracker(values[i, j])
                self.value_trackers[(i, j)] = tracker
                
                # --- 修改处：保留 3 位小数，字号调小至 14 ---
                val_num = DecimalNumber(
                    values[i, j], num_decimal_places=3, show_ellipsis=False,
                    font_size=14, color=WHITE
                ).move_to(pos + np.array([0, -0.2, 0]))
                
                val_num.add_updater(lambda m, r=i, c=j: m.set_value(self.value_trackers[(r, c)].get_value()))
                val_num.set_z_index(10)
                
                grid_group.add(cell, reward_label, val_num)
                
                if rewards[i, j] != 1.0:
                    action_idx = policy[i, j]
                    vec = self.action_vecs[action_idx]
                    arrow = Arrow(
                        start=pos, end=pos + vec * 0.4, buff=0, 
                        color=GOLD, stroke_width=3, 
                        max_tip_length_to_length_ratio=0.4
                    )
                    arrow.set_opacity(0.6) 
                    self.arrows[(i, j)] = arrow
                    grid_group.add(arrow)

        grid_group.move_to(ORIGIN)
        self.play(Create(grid_group))
        self.play(FadeOut(self.title))
        self.play(grid_group.animate.scale(0.8).move_to(LEFT * 3.5))
        
        # --- C. 右侧信息面板 ---
        info_group = VGroup()
        phase_title = Text("Phase: Initialization", font_size=24, color=WHITE).to_edge(RIGHT, buff=1.0).shift(UP*2.5)
        iter_text = Text("Policy Iteration: 0", font_size=20).next_to(phase_title, DOWN, buff=0.3)
        status_text = Text("Ready", font_size=20, color=YELLOW).next_to(iter_text, DOWN, buff=0.3)
        
        # --- Log Scale Axis Construction ---
        y_height = 2.5
        x_width = 3.5
        
        # 轴线
        x_axis = Line(ORIGIN, RIGHT * x_width, color=BLUE)
        y_axis = Line(ORIGIN, UP * y_height, color=BLUE)
        axes_group = VGroup(x_axis, y_axis).next_to(status_text, DOWN, buff=0.8).shift(LEFT * 0.5)
        
        # X轴刻度 (0, 5, 10)
        x_labels = VGroup()
        for i in range(0, 11, 2):
            tick = Line(UP*0.1, DOWN*0.1, color=BLUE).move_to(axes_group[0].get_start() + RIGHT * (i/10) * x_width)
            label = Text(str(i), font_size=12).next_to(tick, DOWN, buff=0.1)
            x_labels.add(tick, label)
            
        # Y轴刻度 (Log Scale: 10^0, 10^-1, 10^-2, 10^-3, 10^-4)
        y_labels = VGroup()
        log_vals = [0, -1, -2, -3, -4]
        labels_tex = ["10^0", "10^{-1}", "10^{-2}", "10^{-3}", "10^{-4}"]
        
        for i, val in enumerate(log_vals):
            # Normalize -4~0 to 0~1
            norm_y = (val - (-4)) / 4
            pos = axes_group[1].get_start() + UP * norm_y * y_height
            tick = Line(LEFT*0.1, RIGHT*0.1, color=BLUE).move_to(pos)
            label = MathTex(labels_tex[i], font_size=16).next_to(tick, LEFT, buff=0.1)
            y_labels.add(tick, label)
            
        # 轴标签
        x_label_text = Text("Step", font_size=16).next_to(x_axis, DOWN, buff=0.4)
        y_label_text = Text("Max Error (Log)", font_size=16, color=WHITE).rotate(90*DEGREES).next_to(y_axis, LEFT, buff=0.6)
        
        # k 值文字
        k_text = Text("Fixed Steps: k=--", color=RED, font_size=20).next_to(x_axis, DOWN, buff=0.6)

        plot_group = VGroup(axes_group, x_labels, y_labels, x_label_text, y_label_text, k_text)
        info_group.add(phase_title, iter_text, status_text, plot_group)
        
        self.play(Write(info_group))
        
        # --- D. 主循环 ---
        policy_stable = False
        outer_iter = 0
        
        # 增加最大迭代次数，确保 Agent 能找到路
        while not policy_stable and outer_iter < 15:
            current_target_steps = self.pe_steps_schedule[min(outer_iter, len(self.pe_steps_schedule)-1)]
            outer_iter += 1
            
            # 更新 k 值文字
            new_k_text = Text(f"Fixed Steps: k={current_target_steps}", color=RED, font_size=20).move_to(k_text)
            
            self.play(
                iter_text.animate.become(Text(f"Policy Iteration: {outer_iter}", font_size=20).move_to(iter_text)),
                Transform(k_text, new_k_text),
                run_time=0.2
            )
            
            # --- Phase 1: Policy Evaluation ---
            self.play(
                phase_title.animate.become(Text("Phase: Policy Evaluation", font_size=24, color=YELLOW).move_to(phase_title)),
                status_text.animate.become(Text(f"Running for k={current_target_steps} steps", font_size=20, color=YELLOW).move_to(status_text)),
                *[a.animate.set_opacity(0.3) for a in self.arrows.values()], 
                run_time=0.5
            )
            
            plot_dots = VGroup()
            last_dot = None
            
            for pe_step in range(1, current_target_steps + 1):
                max_diff = 0
                new_values = values.copy()
                anims = []
                
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if rewards[i, j] == 1.0: continue 

                        action_idx = policy[i, j]
                        di, dj = self.actions[action_idx]
                        ni, nj = i + di, j + dj
                        
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            r_next = rewards[ni, nj]
                            v_next = values[ni, nj]
                        else:
                            r_next = rewards[i, j] 
                            v_next = values[i, j]
                            
                        val = r_next + self.gamma * v_next
                        new_values[i, j] = val
                        
                        diff = abs(val - values[i, j])
                        if diff > max_diff:
                            max_diff = diff
                            
                        if diff > 0.0001:
                            anims.append(self.value_trackers[(i, j)].animate.set_value(val))
                
                values = new_values
                
                # --- 图表打点 (Log Scale) ---
                plot_err = max(max_diff, 1e-4)
                plot_err = min(plot_err, 1.0)
                
                log_err = np.log10(plot_err) # -4 to 0
                norm_y = (log_err - (-4)) / 4
                
                # X轴映射: step 1->0, step 10->width
                x_pos = (pe_step / 10) * x_width
                y_pos = norm_y * y_height
                
                dot_pos = axes_group[0].get_start() + RIGHT * x_pos + UP * y_pos
                new_dot = Dot(dot_pos, color=YELLOW, radius=0.06)
                plot_dots.add(new_dot)
                
                connect_line = None
                if last_dot:
                    connect_line = Line(last_dot.get_center(), new_dot.get_center(), color=YELLOW, stroke_width=2)
                    plot_dots.add(connect_line)
                last_dot = new_dot

                chart_anims = [Create(new_dot)]
                if connect_line: chart_anims.append(Create(connect_line))
                
                if anims:
                    self.play(
                        AnimationGroup(*chart_anims), 
                        LaggedStart(*anims, lag_ratio=0.01), 
                        run_time=0.4
                    )
                else:
                    self.play(AnimationGroup(*chart_anims), run_time=0.4)

            # 变绿
            if last_dot:
                self.play(last_dot.animate.set_color(GREEN).scale(1.5), run_time=0.3)
                self.play(last_dot.animate.scale(1/1.5), run_time=0.1)

            self.play(FadeOut(plot_dots))

            # --- Phase 2: Policy Improvement ---
            self.play(
                phase_title.animate.become(Text("Phase: Policy Improvement", font_size=24, color=GREEN).move_to(phase_title)),
                status_text.animate.become(Text("Greedy Update", font_size=20, color=GREEN).move_to(status_text)),
                *[a.animate.set_opacity(1.0) for a in self.arrows.values()],
                run_time=0.5
            )
            
            policy_stable = True
            change_anims = []
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if rewards[i, j] == 1.0: continue
                    
                    old_action = policy[i, j]
                    
                    q_values = []
                    for idx, (di, dj) in enumerate(self.actions):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            r_next = rewards[ni, nj]
                            v_next = values[ni, nj]
                        else:
                            r_next = rewards[i, j]
                            v_next = values[i, j]
                        
                        q = r_next + self.gamma * v_next
                        q_values.append(q)
                    
                    best_action = np.argmax(q_values)
                    
                    if best_action != old_action:
                        policy_stable = False
                        policy[i, j] = best_action
                        
                        arrow = self.arrows[(i, j)]
                        new_vec = self.action_vecs[best_action]
                        center = self.cells[(i, j)].get_center()
                        
                        new_arrow = Arrow(
                            start=center, end=center + new_vec * 0.4, buff=0, 
                            color=GREEN, stroke_width=3, 
                            max_tip_length_to_length_ratio=0.4
                        )
                        change_anims.append(Transform(arrow, new_arrow))
            
            if change_anims:
                self.play(LaggedStart(*change_anims, lag_ratio=0.05), run_time=1.5)
            else:
                self.play(Flash(status_text, color=GREEN, run_time=0.8))
                
        # --- E. 结束与 Agent 演示 ---
        final_text = Text("Converged! Running Agent...", color=GREEN, font_size=24).next_to(status_text, DOWN)
        self.play(Write(final_text))
        
        self.simulate_agent(policy)

    def simulate_agent(self, policy):
        start_pos = (0, 0)
        curr_r, curr_c = start_pos
        
        start_center = self.cells[start_pos].get_center()
        
        # Z-index 300 确保在最顶层
        agent = Dot(color=BLUE, radius=0.25).move_to(start_center).set_z_index(300)
        path = TracedPath(agent.get_center, stroke_color=BLUE_A, stroke_width=4, stroke_opacity=0.8).set_z_index(299)
        
        self.add(path)
        self.play(FadeIn(agent))
        self.wait(0.5)
        
        steps = 0
        while steps < 20: # 增加步数上限以防万一
            action_idx = policy[curr_r, curr_c]
            di, dj = self.actions[action_idx]
            nr, nc = curr_r + di, curr_c + dj
            
            # 碰撞检测
            hit_wall = False
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                target_pos = self.cells[(nr, nc)].get_center()
                
                # 执行移动
                self.play(agent.animate.move_to(target_pos), run_time=0.6, rate_func=smooth)
                curr_r, curr_c = nr, nc
                
                if (curr_r, curr_c) == (4, 4):
                    self.play(Flash(agent, color=YELLOW, line_length=0.5))
                    break
            else:
                # 撞墙可视化：变红并轻微抖动
                hit_wall = True
                
            if hit_wall:
                self.play(Indicate(agent, color=RED, scale_factor=1.2), run_time=0.5)
            
            steps += 1
        
        self.wait(2)