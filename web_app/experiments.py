import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

def run_cliff_walking_experiment():
    st.title("🧪 Interactive RL Laboratory")
    
    # -------------------------------------------------------
    # 1. Environment Explanation (English) [NEW ADDITION]
    # -------------------------------------------------------
    st.info("""
    ### 🏔️ Scenario Setup: The Cliff Walking Problem
    
    **The World:** A **4x12 grid**. The agent starts at the bottom-left (**S**) and must reach the bottom-right (**G**).
    
    **The Rewards & Penalties:**
    * **Step Cost (-1):** Every move costs **-1 point**. This encourages the agent to find the *fastest* route.
    * **The Cliff (-100):** The bottom row (between S and G) is a cliff. Falling off results in a massive **-100 penalty** and sends the agent back to the start.
    
    **The Dilemma:** The shortest path is right along the edge of the cliff. The safest path is far away.  
    *Does the agent dare to take the risk? It depends on the Discount Factor (Gamma)!*
    """)
    # -------------------------------------------------------

    st.subheader("Experiment Controls")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Hyperparameters")
        # Spec  提到的核心实验变量：Gamma
        gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.9, 
                          help="Low (0.1) = Myopic agent (cares about immediate step cost). High (0.99) = Farsighted agent (fears the future cliff).")
        
        alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.5)
        epsilon = st.slider("Exploration Rate (ε)", 0.0, 1.0, 0.1)
        episodes = st.number_input("Training Episodes", 100, 2000, 500)
        
        run_btn = st.button("🚀 Run Experiment", type="primary")

    with col2:
        tab_path, tab_curve = st.tabs(["🗺️ Final Policy Path", "📈 Reward Curve"])
        with tab_path:
            path_placeholder = st.empty()
        with tab_curve:
            curve_placeholder = st.empty()

    if run_btn:
        # 使用 v1 版本
        env = gym.make('CliffWalking-v1', render_mode="rgb_array")
        q_table = np.zeros([env.observation_space.n, env.action_space.n])
        rewards = []

        progress_bar = st.progress(0)
        status = st.empty()

        # 训练循环
        for i in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Epsilon-Greedy
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state])
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Q-Learning Update
                old_val = q_table[state, action]
                next_max = np.max(q_table[next_state])
                new_val = old_val + alpha * (reward + gamma * next_max - old_val)
                q_table[state, action] = new_val
                
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)

            # UI 更新
            if (i+1) % (episodes // 10) == 0:
                progress_bar.progress((i + 1) / episodes)
                status.text(f"Training Episode {i+1} / {episodes}")
                curve_placeholder.line_chart(rewards)
                time.sleep(0.01)
        
        progress_bar.progress(100)
        status.success("Done!")
        
        # 绘图
        fig = plot_final_path(q_table)
        path_placeholder.pyplot(fig)

def plot_final_path(q_table):
    """独立的绘图函数"""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(0, 13, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制环境元素
    cliff = plt.Rectangle((1, 0), 10, 1, color='#ff6666', alpha=0.3)
    ax.add_patch(cliff)
    ax.text(6, 0.5, "CLIFF (-100)", ha='center', va='center', color='red', fontweight='bold')
    ax.text(0.5, 0.5, "START", ha='center', color='green', fontweight='bold')
    ax.text(11.5, 0.5, "GOAL", ha='center', color='blue', fontweight='bold')

    # 绘制策略箭头
    for s in range(48):
        row = 3 - (s // 12)
        col = s % 12
        if row == 0 and 0 < col < 12: continue
        
        best_action = np.argmax(q_table[s])
        dx, dy = 0, 0
        if best_action == 0: dy = 0.35
        elif best_action == 1: dx = 0.35
        elif best_action == 2: dy = -0.35
        elif best_action == 3: dx = -0.35
        
        # 如果策略是“往悬崖里跳”，标红
        c = 'red' if (row==1 and 0<col<11 and best_action==2) else 'black'
        ax.arrow(col+0.5, row+0.5, dx, dy, head_width=0.1, head_length=0.1, fc=c, ec=c)
    
    return fig