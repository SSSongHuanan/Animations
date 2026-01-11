import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

# ==========================================
# 1. 核心绘图工具
# ==========================================

def plot_heatmap_and_arrows(q_table, shape, env_type="cliff"):
    """
    [V2.0 Final] 热力图 + 策略箭头可视化
    """
    state_values = np.max(q_table, axis=1).reshape(shape)
    rows, cols = shape
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(state_values, annot=False, cmap="YlGnBu", cbar=True, ax=ax, alpha=0.9,
                cbar_kws={'label': 'Estimated State Value (V)'})
    
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            best_action = np.argmax(q_table[s])
            value = state_values[r, c]
            
            # 数值 (左上角)
            ax.text(c + 0.05, r + 0.15, f"{value:.1f}", 
                    color='#333333', fontsize=8, ha='left', va='center')

            # 特殊区域
            is_terminal = False
            center_text_style = {'ha': 'center', 'va': 'center', 'weight': 'bold', 'color': 'white'}
            
            if env_type == "cliff":
                if r == 3 and c == 11: 
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, color='#66cc66', alpha=0.7))
                    ax.text(c+0.5, r+0.5, "G", **center_text_style, fontsize=14)
                    is_terminal = True
                elif r == 3 and 0 < c < 11: 
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, color='#ff4444', alpha=0.6))
                    ax.text(c+0.5, r+0.5, "☠️", ha='center', va='center', fontsize=12)
                    continue 
                elif r == 3 and c == 0: 
                    ax.text(c+0.5, r+0.5, "S", ha='center', va='center', weight='bold', color='black', fontsize=12)

            elif env_type == "frozen":
                desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
                char = desc[r][c]
                if char == 'H': 
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, color='#222222', alpha=0.6))
                    continue
                elif char == 'G': 
                    ax.text(c+0.5, r+0.5, "🏆", ha='center', va='center', fontsize=18)
                    is_terminal = True
                elif char == 'S': 
                    ax.text(c+0.5, r+0.5, "S", ha='center', va='center', weight='bold', color='black')

            # 策略箭头
            if not is_terminal:
                arrow_len = 0.25
                dx, dy = 0, 0
                if env_type == "cliff": 
                    if best_action == 0: dy = -arrow_len
                    elif best_action == 1: dx = arrow_len
                    elif best_action == 2: dy = arrow_len
                    elif best_action == 3: dx = -arrow_len
                elif env_type == "frozen": 
                    if best_action == 0: dx = -arrow_len
                    elif best_action == 1: dy = arrow_len
                    elif best_action == 2: dx = arrow_len
                    elif best_action == 3: dy = -arrow_len
                
                ax.arrow(c + 0.5, r + 0.5, dx, dy, 
                         head_width=0.08, head_length=0.08, fc='black', ec='black', alpha=0.6)

    ax.set_title(f"Learned Policy (Arrows) & State Values (Numbers) - {env_type.capitalize()}", fontsize=14, pad=10)
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    return fig

def plot_learning_curve(data, title, ylabel, xlabel="Episode", optimal_line=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data, linewidth=1.5, label="Agent Performance")
    
    if optimal_line is not None:
        ax.axhline(y=optimal_line, color='r', linestyle='--', alpha=0.6, label=f"Optimal ({optimal_line})")
        ax.legend()
        
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig

# ==========================================
# 2. 训练逻辑
# ==========================================

def train_q_learning(env, episodes, alpha, gamma, epsilon, render_placeholder=None):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_history = []
    steps_history = []
    
    if render_placeholder:
        chart_container = render_placeholder.container()
        status_text = chart_container.empty()
        live_chart = chart_container.empty()
    
    progress_bar = st.progress(0)

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 200: 
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            old_val = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_val = old_val + alpha * (reward + gamma * next_max - old_val)
            q_table[state, action] = new_val
            
            state = next_state
            total_reward += reward
            steps += 1
            
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        # 实时更新
        if (i+1) % (max(1, episodes // 20)) == 0:
            progress_bar.progress((i + 1) / episodes)
            if render_placeholder:
                status_text.write(f"⏳ Training... Episode {i+1}/{episodes}")
                
                # [关键修复] 使用 min_periods=1
                # 这意味着：如果数据不够20个，有多少算多少的平均值。
                # 这样曲线从第1个点开始就是平滑计算的，不会在第20个点突然变样。
                df = pd.DataFrame({"Reward (Mov Avg)": rewards_history})
                live_chart.line_chart(df.rolling(window=20, min_periods=1).mean())
                
    progress_bar.progress(100)
    if render_placeholder:
        status_text.success("Training Complete!")
    return q_table, rewards_history, steps_history

# ==========================================
# 3. 实验模块 1: Cliff Walking
# ==========================================
def run_cliff_walking_experiment():
    st.header("🏔️ Experiment 1: Cliff Walking")
    st.info("""
    **Concept:** Risk vs. Reward (Gamma).
    * **Low Gamma (0.1):** Myopic. The agent risks walking on the edge because it's the shortest path.
    * **High Gamma (0.99):** Farsighted. The agent takes a longer, safer detour to avoid the -100 penalty.
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Hyperparameters")
        gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.99, key="cw_gamma")
        alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.5, key="cw_alpha")
        epsilon = st.slider("Exploration (ε)", 0.0, 1.0, 0.1, key="cw_eps")
        episodes = st.number_input("Episodes", 10, 2000, 500, key="cw_ep")
        start_btn = st.button("Train Agent", type="primary", key="cw_btn")

    with col2:
        vis_container = st.container()

    if start_btn:
        env = gym.make('CliffWalking-v1')
        with vis_container:
            q_table, rewards, steps = train_q_learning(env, episodes, alpha, gamma, epsilon, render_placeholder=vis_container)
            
            st.divider()
            tab1, tab2 = st.tabs(["🗺️ Interpretation (Heatmap)", "📈 Performance (Metrics)"])
            
            with tab1:
                st.write("### State-Value & Policy")
                st.caption("Visualization of the agent's strategy.")
                fig = plot_heatmap_and_arrows(q_table, (4, 12), env_type="cliff")
                st.pyplot(fig)
            
            with tab2:
                # [关键修复] 静态图表也使用 min_periods=1 保持一致
                st.write("### 1. Average Reward per Episode")
                smooth_rewards = pd.Series(rewards).rolling(window=20, min_periods=1).mean()
                fig_reward = plot_learning_curve(
                    smooth_rewards, 
                    title=f"Reward Convergence (Smoothed, Window=20)", 
                    ylabel="Total Reward",
                    optimal_line=-13
                )
                st.pyplot(fig_reward)
                st.info("""
                * **-100 zone:** Falling off the cliff.
                * **-13 line:** Optimal path found.
                """)
                
                st.write("### 2. Steps to Goal")
                smooth_steps = pd.Series(steps).rolling(window=20, min_periods=1).mean()
                fig_steps = plot_learning_curve(
                    smooth_steps,
                    title="Steps per Episode",
                    ylabel="Number of Steps"
                )
                st.pyplot(fig_steps)

# ==========================================
# 4. 实验模块 2: Frozen Lake
# ==========================================
def run_frozen_lake_experiment():
    st.header("❄️ Experiment 2: Frozen Lake")
    st.warning("""
    **Concept:** Stochastic Environments (Uncertainty).
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Hyperparameters")
        is_slippery = st.checkbox("Enable Slippery Ice?", value=True)
        gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.95, key="fl_gamma")
        alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.8, key="fl_alpha")
        epsilon = st.slider("Exploration (ε)", 0.0, 1.0, 0.1, key="fl_eps")
        episodes = st.number_input("Episodes", 10, 5000, 2000, key="fl_ep")
        start_btn = st.button("Train Agent", type="primary", key="fl_btn")

    with col2:
        vis_container = st.container()

    if start_btn:
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery)
        with vis_container:
            q_table, rewards, steps = train_q_learning(env, episodes, alpha, gamma, epsilon, render_placeholder=vis_container)
            
            st.divider()
            tab1, tab2 = st.tabs(["🗺️ Policy Visualization", "📊 Success Analysis"])
            
            with tab1:
                st.write("### Learned Policy")
                fig = plot_heatmap_and_arrows(q_table, (4, 4), env_type="frozen")
                st.pyplot(fig)
            
            with tab2:
                # Frozen Lake 默认窗口大一点，但也加上 min_periods 防止小 episode 报错
                win_size = 50
                st.write(f"### Success Rate (Window={win_size})")
                
                success_rate = pd.Series(rewards).rolling(window=win_size, min_periods=1).mean()
                fig_success = plot_learning_curve(
                    success_rate,
                    title="Moving Average Success Rate",
                    ylabel="Success Probability (0.0 - 1.0)",
                    optimal_line=0.7 if is_slippery else 1.0
                )
                st.pyplot(fig_success)

# ==========================================
# 5. 主入口
# ==========================================
def run_experiments_module():
    st.title("🧪 Interactive RL Laboratory")
    exp_type = st.selectbox(
        "Choose an Experiment:",
        ["1. Cliff Walking (Deterministic Risk)", "2. Frozen Lake (Stochastic Uncertainty)"]
    )
    if "Cliff" in exp_type:
        run_cliff_walking_experiment()
    elif "Frozen" in exp_type:
        run_frozen_lake_experiment()