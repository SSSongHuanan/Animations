from manim import *
import numpy as np
import random

class DQNDemo(Scene):
    def construct(self):
        # --- 0. 全局配置 ---
        self.grid_size = 3
        self.cell_size = 1.2 # 稍微缩小以便放下更多东西
        self.grid_spacing = 1.3
        
        # --- 1. 理论介绍 ---
        self.play_intro()
        
        # --- 2. 核心演示 ---
        self.play_dqn_process()

    def play_intro(self):
        title = Text("Deep Q-Network (DQN)", font_size=48, color=BLUE).to_edge(UP)
        subtitle = Text("From Table to Neural Network", font_size=32, color=GREY).next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle))
        
        # 对比图示
        # 左边：Q-Table
        table_viz = VGroup(
            Rectangle(height=2, width=2, grid_xstep=0.5, grid_ystep=0.5),
            Text("Q-Table", font_size=24).next_to(Rectangle(height=2, width=2), DOWN)
        ).shift(LEFT * 3)
        
        # 右边：Neural Net
        nn_viz = VGroup(
            Circle(radius=0.5, color=BLUE).set_fill(BLUE, 0.5),
            Text("Neural Net", font_size=24).next_to(Circle(radius=0.5), DOWN)
        ).shift(RIGHT * 3)
        
        arrow = Arrow(table_viz.get_right(), nn_viz.get_left(), buff=0.5)
        cross = Cross(table_viz).set_color(RED)
        
        self.play(Create(table_viz))
        self.wait(0.5)
        self.play(Create(cross))
        self.play(GrowArrow(arrow), FadeIn(nn_viz))
        
        explanation = Text("Approximate Q(s, a) with Weights", font_size=24, color=YELLOW).next_to(arrow, UP)
        self.play(Write(explanation))
        self.wait(2)
        
        self.play(
            FadeOut(table_viz), FadeOut(cross), FadeOut(arrow), FadeOut(nn_viz), FadeOut(explanation),
            FadeOut(subtitle),
            title.animate.scale(0.8)
        )

    def create_neural_net(self):
        """创建一个可视化的神经网络 VGroup"""
        layers = [2, 5, 4] # 输入层(x,y), 隐藏层, 输出层(Q-values)
        
        neurons = VGroup()
        edges = VGroup()
        
        # 垂直间距
        layer_x_spacing = 1.5
        neuron_y_spacing = 0.6
        
        layer_groups = []
        
        for i, layer_size in enumerate(layers):
            layer_group = VGroup()
            x = i * layer_x_spacing
            
            # 居中对齐 y
            y_start = (layer_size - 1) * neuron_y_spacing / 2
            
            for j in range(layer_size):
                y = y_start - j * neuron_y_spacing
                circle = Circle(radius=0.12, color=WHITE, stroke_width=2, fill_color=BLACK, fill_opacity=1)
                circle.move_to([x, y, 0])
                layer_group.add(circle)
            
            layer_groups.append(layer_group)
            neurons.add(layer_group)
            
        # 创建连接线
        for i in range(len(layers) - 1):
            curr_layer = layer_groups[i]
            next_layer = layer_groups[i+1]
            
            for n1 in curr_layer:
                for n2 in next_layer:
                    edge = Line(n1.get_center(), n2.get_center(), stroke_width=1, color=GREY_C, stroke_opacity=0.5)
                    edges.add(edge)
                    
        # 整体居中
        net_group = VGroup(edges, neurons)
        net_group.move_to(ORIGIN)
        
        # 标签
        input_label = Text("Input\n(State)", font_size=16, color=BLUE).next_to(layer_groups[0], UP)
        output_label = Text("Output\n(Q-Values)", font_size=16, color=GREEN).next_to(layer_groups[-1], UP)
        
        return net_group, layer_groups, edges, VGroup(input_label, output_label)

    def play_dqn_process(self):
        # --- A. 布局初始化 ---
        
        # 1. 左侧：网格世界
        rewards = np.array([
            [-0.04, -0.5, -0.04],
            [-0.04, -1.0,  -0.04],
            [-0.04, -0.04,  1.0]
        ])
        
        grid_group = VGroup()
        cells = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = np.array([(j - 1) * self.grid_spacing, (1 - i) * self.grid_spacing, 0])
                
                color = BLACK
                if rewards[i,j] == 1.0: color = GREEN_E
                elif rewards[i,j] == -1.0: color = RED_E
                elif rewards[i,j] == -0.5: color = ORANGE
                
                cell = Square(side_length=self.cell_size, fill_color=color, fill_opacity=0.5, color=WHITE)
                cell.move_to(pos)
                cells[(i,j)] = cell
                
                # Label
                if rewards[i,j] == 1.0: txt = "+1"
                elif rewards[i,j] == -1.0: txt = "-1"
                elif rewards[i,j] == -0.5: txt = "Mud"
                else: txt = ""
                if txt:
                    t = Text(txt, font_size=16).move_to(pos)
                    grid_group.add(cell, t)
                else:
                    grid_group.add(cell)
                    
        grid_group.move_to(ORIGIN).to_edge(LEFT, buff=0.5)
        
        # 2. 中间：神经网络
        net_viz, layer_groups, edges, labels = self.create_neural_net()
        # 将网络放在 Grid 和右边界的中间
        center_x = (grid_group.get_right()[0] + config.frame_width/2) / 2
        # 其实可以直接放中间，然后把 Bar Chart 挤到最右
        net_viz.move_to([0, -0.5, 0])
        labels[0].next_to(layer_groups[0], UP)
        labels[1].next_to(layer_groups[-1], UP)
        
        net_group_all = VGroup(net_viz, labels)
        
        # 3. 右侧：Q值柱状图
        # 手动画 Bar Chart 因为 Manim 的 BarChart 有时候不好控制位置
        bar_names = ["U", "D", "L", "R"]
        bars = VGroup()
        bar_values_text = VGroup()
        bar_bg_group = VGroup()
        
        x_start = 4.5
        y_start = -1.5
        bar_width = 0.4
        bar_spacing = 0.6
        
        for i in range(4):
            x = x_start + i * bar_spacing
            # 背景框
            bg = Rectangle(width=bar_width, height=3, stroke_color=GREY, stroke_width=1).move_to([x, 0, 0])
            bar_bg_group.add(bg)
            
            # 柱子 (初始高度0)
            bar = Rectangle(width=bar_width, height=0.1, fill_color=GREEN, fill_opacity=0.8, stroke_width=0)
            bar.move_to(bg.get_bottom() + UP * 0.05)
            bars.add(bar)
            
            # 标签
            lbl = Text(bar_names[i], font_size=20).next_to(bg, DOWN)
            bar_bg_group.add(lbl)
            
        chart_title = Text("Q-Values", font_size=24, color=YELLOW).next_to(bar_bg_group, UP)
        chart_group = VGroup(chart_title, bar_bg_group, bars)
        
        # 动画显示所有组件
        self.play(
            FadeIn(grid_group),
            FadeIn(net_group_all),
            FadeIn(chart_group)
        )
        
        # --- B. 演示过程 ---
        
        agent = Dot(color=BLUE_A, radius=0.15).set_z_index(10)
        curr_r, curr_c = 0, 0
        start_cell = cells[(0,0)]
        agent.move_to(start_cell.get_center())
        self.play(FadeIn(agent))
        
        # 伪造一些数据来演示训练过程
        # Path: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2)
        # 每个步骤：
        # 1. State Input 动画
        # 2. Forward Pass 动画
        # 3. Bar Chart Update 动画
        # 4. Agent Move 动画
        # 5. Reward & Backprop 动画
        
        steps = [
            # ((r,c), action_idx, reward, q_values_list)
            # Action: 0:U, 1:D, 2:L, 3:R
            # 初始状态，网络瞎猜
            ((0,0), 1, -0.04, [0.1, 0.2, 0.1, 0.15]), 
            ((1,0), 1, -0.04, [0.2, 0.5, 0.1, 0.2]), 
            ((2,0), 3, -0.04, [0.1, 0.1, 0.2, 0.6]),
            ((2,1), 3,  1.0,  [0.1, 0.0, 0.3, 0.9])
        ]
        
        for idx, (pos, action_idx, reward, q_vals) in enumerate(steps):
            r, c = pos
            
            # 1. 高亮当前状态
            current_cell = cells[(r, c)]
            hl = current_cell.animate.set_stroke(YELLOW, width=4)
            self.play(hl, run_time=0.3)
            
            # 2. 前向传播动画 (Forward Pass)
            # 粒子从左流向右
            run_time = 1.0
            
            # 激活输入层
            input_flash = [n.animate.set_fill(YELLOW) for n in layer_groups[0]]
            self.play(*input_flash, run_time=0.2)
            
            # 信号通过连线
            edge_anims = []
            for edge in edges:
                edge_anims.append(ShowPassingFlash(edge.copy().set_color(YELLOW), time_width=0.2))
            self.play(LaggedStart(*edge_anims, lag_ratio=0.1, run_time=run_time))
            
            # 激活输出层
            output_flash = [n.animate.set_fill(GREEN) for n in layer_groups[-1]]
            self.play(*output_flash, run_time=0.2)
            
            # 恢复神经元颜色
            restore = [n.animate.set_fill(BLACK) for n in layer_groups[0]] + [n.animate.set_fill(BLACK) for n in layer_groups[-1]]
            self.play(*restore, run_time=0.1)
            
            # 3. 更新右侧柱状图
            bar_anims = []
            max_h = 2.5
            for b_idx, val in enumerate(q_vals):
                target_h = val * max_h
                # 重新定位柱子
                new_bar = Rectangle(width=bar_width, height=target_h, fill_color=GREEN, fill_opacity=0.8, stroke_width=0)
                # 底部对齐
                bg = bar_bg_group[b_idx*2] # bg rect
                new_bar.move_to(bg.get_bottom() + UP * (target_h/2 + 0.05))
                
                # 如果是选中的动作，高亮
                if b_idx == action_idx:
                    new_bar.set_color(YELLOW)
                
                bar_anims.append(Transform(bars[b_idx], new_bar))
            
            self.play(*bar_anims)
            
            # 4. 执行动作
            move_vec = [UP, DOWN, LEFT, RIGHT][action_idx]
            # 计算下一个格子的位置 (为了简单直接用 idx+1 的数据，如果是最后一步则进终点)
            if idx < len(steps) - 1:
                next_pos_tuple = steps[idx+1][0]
                target_cell = cells[next_pos_tuple]
            else:
                # 进终点 (2,2)
                target_cell = cells[(2,2)]
                
            self.play(agent.animate.move_to(target_cell.get_center()))
            
            # 5. 反向传播 (Backpropagation)
            # 显示 Reward
            rew_lbl = Text(f"Reward: {reward}", color=GOLD, font_size=24).next_to(chart_group, DOWN)
            self.play(Write(rew_lbl))
            
            # 计算 Loss (视觉演示)
            # 目标 Q (Target) 假设比当前预测高一点
            target_q_val = q_vals[action_idx] + 0.2 
            
            # 在柱状图上显示“目标值”虚线
            target_line = DashedLine(
                start=bars[action_idx].get_left(), 
                end=bars[action_idx].get_right()
            ).shift(UP * 0.3) # 往上一点
            
            loss_text = Text("Loss", color=RED, font_size=20).next_to(target_line, UP)
            
            self.play(Create(target_line), FadeIn(loss_text))
            self.wait(0.5)
            
            # 红色脉冲回传
            back_anims = []
            # 翻转 edges 列表或者只是视觉效果
            # 这里简单起见，让 edges 变红闪烁
            self.play(
                *[e.animate.set_color(RED) for e in edges], 
                run_time=0.2
            )
            self.play(
                *[e.animate.set_color(GREY_C) for e in edges],
                run_time=0.3
            )
            
            self.play(
                FadeOut(rew_lbl), FadeOut(target_line), FadeOut(loss_text),
                current_cell.animate.set_stroke(WHITE, width=0) # 移除高亮
            )

        # --- C. 训练完成 ---
        self.play(
            FadeOut(agent),
            FadeOut(bar_bg_group), FadeOut(bars), FadeOut(chart_title),
            net_group_all.animate.scale(0.8).shift(UP * 2) # 把网络移上去一点
        )
        
        final_text = Text("Training Complete: Optimal Policy", color=GREEN, font_size=32).move_to(DOWN * 2)
        self.play(Write(final_text))
        
        # 快速跑一遍最优路径
        agent.move_to(cells[(0,0)].get_center())
        self.play(FadeIn(agent))
        path = TracedPath(agent.get_center, stroke_color=BLUE, stroke_width=4)
        self.add(path)
        
        waypoints = [(1,0), (2,0), (2,1), (2,2)]
        for wp in waypoints:
            self.play(agent.animate.move_to(cells[wp].get_center()), run_time=0.4)
            
        self.play(Flash(agent, color=YELLOW))
        self.wait(2)