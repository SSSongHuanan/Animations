import streamlit as st
import os

# ==========================================
# 1. Utility Function
# ==========================================
def get_manim_video(scene_folder, video_name=None, quality="1080p60"):
    if video_name is None:
        video_name = scene_folder
    # Path structure: media/videos/{Folder}/1080p60/{Filename}.mp4
    path = os.path.join("media", "videos", scene_folder, quality, f"{video_name}.mp4")
    if os.path.exists(path):
        return path
    return None

# ==========================================
# 2. Educational Content Database
# ==========================================
ANIMATION_DATA = {
    "Policy Iteration": {
        "folder": "Policy_iteration",
        "file": "PolicyIteration",
        "title": "Policy Iteration",
        "description": """
        **Policy Iteration** is a classic **Model-Based Dynamic Programming** method. It decomposes the learning process into two alternating phases:
        1. **Policy Evaluation:** Calculates the state-value function $V(s)$ for the current policy.
        2. **Policy Improvement:** Greedily updates the policy $\pi$ based on the new $V(s)$.
        This cycle repeats until the policy stabilizes (converges).
        """,
        "latex": r"V(s) \leftarrow \sum_{s', r} p(s', r|s, \pi(s)) [r + \gamma V(s')]",
        "highlights": [
            "Observe that the arrows (Policy) remain static during the Evaluation phase, while only the colors (Value) update.",
            "Notice the sharp change in arrow direction during the Improvement phase.",
            "The agent eventually finds the optimal path avoiding the traps."
        ]
    },
    "Value Iteration": {
        "folder": "Value_iteration",
        "file": "ValueIterationGeneral",
        "title": "Value Iteration",
        "description": """
        **Value Iteration** combines evaluation and improvement into a single step. Instead of maintaining an explicit policy during the loop, it iteratively updates the value function $V(s)$.
        
        The optimal policy is derived directly from the converged value function. This method is often more computationally efficient than Policy Iteration.
        """,
        "latex": r"V_{k+1}(s) \leftarrow \max_a \sum_{s', r} p(s', r|s, a) [r + \gamma V_k(s')]",
        "highlights": [
            "Pay attention to the **'Ripple Effect'**: values propagate outward from the Goal state.",
            "Darker colors represent higher values (closer to the goal).",
            "Unlike Policy Iteration, the policy is not explicit until the end."
        ]
    },
    "Q-Learning": {
        "folder": "QLearning",
        "file": "QLearningDemo",
        "title": "Q-Learning (Off-Policy)",
        "description": """
        **Q-Learning** is the most popular **Model-Free** algorithm. It learns the Action-Value function $Q(s,a)$ directly from experience.
        
        It is an **Off-Policy** algorithm because it updates its Q-values assuming the greedy action is taken (max Q), even if the agent is actually exploring (random action).
        """,
        "latex": r"Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s,a)]",
        "highlights": [
            "Watch how the Q-values (colors) drop rapidly when the agent falls into the cliff.",
            "Notice that even when the agent takes a random path (Exploration), the Q-value update targets the best possible future.",
            "The visualization shows the transition from a 'Tabula Rasa' (blank slate) to a learned policy."
        ]
    },
    "SARSA": {
        "folder": "SARSA",
        "file": "SARSADemo",
        "title": "SARSA (On-Policy)",
        "description": """
        **SARSA** (State-Action-Reward-State-Action) is a **Model-Free, On-Policy** algorithm.
        
        Unlike Q-Learning, SARSA is 'conservative'. When updating Q-values, it considers the action it *actually* took next (which might be a random mistake). This makes it learn safer paths in dangerous environments.
        """,
        "latex": r"Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s', a') - Q(s,a)]",
        "highlights": [
            "Compare this with Q-Learning: SARSA often learns a path further away from the cliff.",
            "This happens because SARSA 'fears' the random probability of falling during exploration.",
            "Key Mathematical Difference: The max operator is replaced by the actual next action."
        ]
    },
    "Deep Q-Network (DQN)": {
        "folder": "DQN",
        "file": "DQNDemo",
        "title": "Deep Q-Network (DQN)",
        "description": """
        When the state space is too large for a table (e.g., video game pixels), we use a **Neural Network** to approximate the Q-function.
        
        DQN introduces **Experience Replay** and **Target Networks** to stabilize training, enabling RL to solve complex, high-dimensional problems.
        """,
        "latex": r"L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]",
        "highlights": [
            "Visualize the Neural Network replacing the Q-Table.",
            "Observe how experiences are stored in the Replay Buffer and sampled in batches.",
            "This represents the foundation of modern Deep Reinforcement Learning."
        ]
    }
}

# ==========================================
# 3. Page Rendering Logic
# ==========================================
def show_animation_library():
    st.title("🎥 Algorithm Animation Library")
    st.markdown("Visualize the abstract mechanisms of **Value Propagation** and **Policy Updates** through programmatic animations.")
    
    # 1. Sidebar Selection
    selected_key = st.sidebar.radio(
        "📖 Select Topic:",
        list(ANIMATION_DATA.keys())
    )
    
    # Get Data
    data = ANIMATION_DATA[selected_key]
    video_path = get_manim_video(data["folder"], data["file"])

    # 2. Main Content
    st.header(data["title"])
    
    # --- Video Player ---
    if video_path:
        st.video(video_path)
    else:
        st.error(f"❌ Video file not found: {data['folder']}/{data['file']}")
        st.info("Tip: Please verify the file name in 'media/videos/'.")

    # --- Educational Content (Tabs) ---
    st.divider()
    
    tab_intro, tab_math = st.tabs(["📝 Concept Introduction", "➗ Mathematical Core"])
    
    with tab_intro:
        st.markdown(data["description"])
        
        if "highlights" in data:
            st.info("**👀 Key Visual Highlights:**\n\n" + 
                    "\n".join([f"* {item}" for item in data["highlights"]]))

    with tab_math:
        st.markdown("#### The Bellman Update Rule")
        st.markdown("The animation programmatically visualizes the dynamics of this equation:")
        st.latex(data["latex"])
        st.caption("Symbols: $s$=State, $a$=Action, $r$=Reward, $\gamma$=Discount Factor, $\\alpha$=Learning Rate")