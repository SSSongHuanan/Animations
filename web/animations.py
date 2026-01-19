import streamlit as st
import os
from typing import Optional, List, Dict, Any
import re

# ==========================================
# 1. Utility Functions
# ==========================================
def _score_quality(q: str):
    """Prefer higher resolution/fps labels first (e.g., 1080p60)."""
    m = re.search(r"(\d+)p(\d+)", q)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d+)p", q)
    if m:
        return (int(m.group(1)), 0)
    return (0, 0)

def pick_best_quality(scene_folder: str, video_name: str) -> Optional[str]:
    """Auto-pick the best available quality folder for a given scene/video."""
    base_dir = os.path.join("media", "videos", scene_folder)
    if not os.path.isdir(base_dir):
        return None

    qualities: List[str] = []
    for q in os.listdir(base_dir):
        mp4 = os.path.join(base_dir, q, f"{video_name}.mp4")
        if os.path.exists(mp4):
            qualities.append(q)

    if not qualities:
        return None

    qualities.sort(key=_score_quality, reverse=True)
    return qualities[0]

def get_manim_video(scene_folder: str, video_name: str, quality: Optional[str] = None) -> Optional[str]:
    if quality is None:
        quality = pick_best_quality(scene_folder, video_name)
    if quality is None:
        return None
    path = os.path.join("media", "videos", scene_folder, quality, f"{video_name}.mp4")
    return path if os.path.exists(path) else None

def render_derivation_steps(steps: List[Dict[str, Any]]):
    """Render derivation as 'real derivation': text paragraphs + separate LaTeX blocks."""
    for i, step in enumerate(steps, start=1):
        title = step.get("title")
        text = step.get("text")
        latex = step.get("latex")
        # number each step for clarity
        if title:
            st.markdown(f"**Step {i}: {title}**")
        else:
            st.markdown(f"**Step {i}**")
        if text:
            st.markdown(text)
        if latex:
            # allow single string or list of latex blocks
            if isinstance(latex, list):
                for block in latex:
                    st.latex(block)
            else:
                st.latex(latex)
        if i != len(steps):
            st.markdown("---")

# ==========================================
# 2. Educational Content Database
# ==========================================
ANIMATION_DATA: Dict[str, Dict[str, Any]] = {
    "Policy Iteration": {
        "folder": "Policy_iteration",
        "file": "PolicyIteration",
        "title": "Policy Iteration",
        "description": """
**Policy Iteration** is a model-based dynamic programming method that alternates:

1) **Policy Evaluation**: compute the value function under the current policy

2) **Policy Improvement**: update the policy greedily with respect to that value function

Repeat until the policy stops changing.
""",
        "latex": r"V^\pi(s)=\sum_{s',r} p(s',r\mid s,\pi(s))\,[r + \gamma V^\pi(s')]",
        "highlights": [
            "During Evaluation: arrows (policy) stay fixed, values update.",
            "During Improvement: arrows change to become greedier.",
            "Repeat until arrows stop changing (convergence)."
        ],
        "derivation_steps": [
            {
                "title": "Start from Bellman expectation equation (fixed policy)",
                "text": "For a fixed policy π, the value equals expected return following π.",
                "latex": r"V^\pi(s)=\mathbb{E}\big[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s, A_t=\pi(s)\big]",
            },
            {
                "title": "Expand the expectation using transition dynamics",
                "text": "Convert the expectation into a sum over next state and reward using the known model p(s',r|s,a).",
                "latex": r"V^\pi(s)=\sum_{s',r} p(s',r\mid s,\pi(s))\,[r + \gamma V^\pi(s')]",
            },
            {
                "title": "Policy improvement step",
                "text": "Given V^π, choose actions that maximize one-step lookahead return; this defines an improved policy.",
                "latex": r"\pi_{new}(s)\in\arg\max_a\sum_{s',r} p(s',r\mid s,a)\,[r+\gamma V^\pi(s')]",
            },
        ],
    },
    "Value Iteration": {
        "folder": "Value_iteration",
        "file": "ValueIterationGeneral",
        "title": "Value Iteration",
        "description": """
**Value Iteration** applies the Bellman optimality backup directly to the value function.

When V converges, the greedy policy extracted from V is optimal.
""",
        "latex": r"V_{k+1}(s)\leftarrow\max_a\sum_{s',r} p(s',r\mid s,a)\,[r+\gamma V_k(s')]",
        "highlights": [
            "Values propagate outward from goal/terminal states (ripple effect).",
            "The greedy action becomes clearer as V stabilizes.",
            "Policy is derived after (or during late) iterations."
        ],
        "derivation_steps": [
            {
                "title": "Optimal value relates to optimal action-value",
                "text": "For the optimal policy, the best action at s achieves V*(s).",
                "latex": r"V^*(s)=\max_a Q^*(s,a)",
            },
            {
                "title": "One-step lookahead definition of Q*",
                "text": "Action-value equals expected immediate reward plus discounted optimal value of the next state.",
                "latex": r"Q^*(s,a)=\sum_{s',r} p(s',r\mid s,a)\,[r+\gamma V^*(s')]",
            },
            {
                "title": "Substitute Q* into V* (Bellman optimality equation)",
                "text": "Combine the two equations to obtain the optimality backup used by value iteration.",
                "latex": r"V^*(s)=\max_a\sum_{s',r} p(s',r\mid s,a)\,[r+\gamma V^*(s')]",
            },
            {
                "title": "Iterative application",
                "text": "Value iteration repeatedly applies the backup operator until changes are small.",
                "latex": r"V_{k+1}=\mathcal{T}^*V_k",
            },
        ],
    },
    "Q-Learning": {
        "folder": "QLearning",
        "file": "QLearningDemo",
        "title": "Q-Learning (Off-Policy)",
        "description": """
**Q-Learning** learns Q(s,a) from experience and is **off-policy** because its target uses a greedy max over next actions.
""",
        "latex": r"Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]",
        "highlights": [
            "Update target uses the best future action (the max).",
            "Even with exploration, learning pushes toward greedy behavior.",
            "In risky tasks, Q-learning may learn aggressive shortest paths."
        ],
        "derivation_steps": [
            {
                "title": "Define TD error (off-policy target)",
                "text": "Q-learning update is a gradient-like step driven by a TD error δ.",
                "latex": [
                    r"\delta = r + \gamma\max_{a'}Q(s',a') - Q(s,a)",
                    r"Q(s,a)\leftarrow Q(s,a)+\alpha\,\delta",
                ],
            },
            {
                "title": "Why it is off-policy",
                "text": "Even if the behavior policy explores, the target assumes the greedy next action (the max).",
                "latex": r"\text{target uses }\max_{a'}Q(s',a')\ \text{even if }a'\sim\varepsilon\text{-greedy}",
            },
        ],
    },
    "SARSA": {
        "folder": "SARSA",
        "file": "SARSADemo",
        "title": "SARSA (On-Policy)",
        "description": """
**SARSA** is on-policy TD control: it updates using the next action actually taken under the current behavior policy.
""",
        "latex": r"Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma Q(s',a')-Q(s,a)]",
        "highlights": [
            "Targets the next sampled action a′ (no max operator).",
            "Under ε-greedy exploration, SARSA tends to learn safer policies.",
            "Great to compare with Q-learning in risky environments."
        ],
        "derivation_steps": [
            {
                "title": "Define TD error (on-policy target)",
                "text": "SARSA uses the next action actually taken a′ to form the TD target.",
                "latex": [
                    r"\delta = r + \gamma Q(s',a') - Q(s,a)",
                    r"Q(s,a)\leftarrow Q(s,a)+\alpha\,\delta",
                ],
            },
            {
                "title": "Contrast with Q-learning",
                "text": "Replace the max over actions with the sampled next action; exploration risk is included in the target.",
                "latex": r"\text{Q-learning uses }\max_{a'}Q(s',a')\quad\text{SARSA uses }Q(s',a')",
            },
        ],
    },
    "Deep Q-Network (DQN)": {
        "folder": "DQN",
        "file": "DQNDemo",
        "title": "Deep Q-Network (DQN)",
        "description": """
**DQN** approximates Q(s,a) with a neural network when tables are infeasible.
It stabilizes learning via **experience replay** and a **target network**.
""",
        "latex": r"L(\theta)=\mathbb{E}\Big[(r+\gamma\max_{a'}Q(s',a';\theta^-)-Q(s,a;\theta))^2\Big]",
        "highlights": [
            "Neural net replaces the Q-table (function approximation).",
            "Replay buffer helps stabilize learning.",
            "Target network parameters θ⁻ lag behind θ."
        ],
        "derivation_steps": [
            {
                "title": "Start from TD target y",
                "text": "DQN uses a target value y computed from the reward and the next-state greedy action.",
                "latex": r"y = r + \gamma\max_{a'}Q(s',a';\theta^-)",
            },
            {
                "title": "Define squared error loss",
                "text": "Train the network to make Q(s,a;θ) match the target y (MSE over samples).",
                "latex": r"L(\theta) = \mathbb{E}\big[(y - Q(s,a;\theta))^2\big]",
            },
            {
                "title": "Why use a target network θ⁻",
                "text": "Keep the target relatively fixed for a while to reduce the moving-target problem during training.",
                "latex": r"\theta^-\ \text{is updated periodically from}\ \theta",
            },
        ],
    },
}

# ==========================================
# 3. Page Rendering Logic (Simplified UI)
# ==========================================
def show_animation_library():
    st.header("🎥 Algorithm Animations")
    st.caption("Manim-based videos that visualize how values and policies evolve during learning.")

    st.sidebar.markdown("## 🎞️ Select a Video")
    keys = list(ANIMATION_DATA.keys())
    selected_key = st.sidebar.radio("Algorithm", keys)
    data = ANIMATION_DATA[selected_key]

    st.subheader(data["title"])

    left, right = st.columns([1.45, 1.0], gap="large")

    with left:
        quality = pick_best_quality(data["folder"], data["file"])
        video_path = get_manim_video(data["folder"], data["file"], quality=quality)

        if video_path:
            st.video(video_path)
        else:
            st.error("Video file not found.")
            st.info("Tip: check your Manim output folder under 'media/videos/'.")
            st.code(os.path.join("media", "videos", data["folder"]), language="text")

    with right:
        with st.container(border=True):
            st.markdown("#### What to look for")
            if data.get("highlights"):
                st.markdown("\n".join([f"- {item}" for item in data["highlights"]]))
            else:
                st.caption("No highlight notes provided yet.")

        with st.container(border=True):
            st.markdown("#### Mathematical core")
            st.latex(data["latex"])
            st.write("Symbols: s=state, a=action, r=reward, γ=discount, α=learning rate, ε=exploration")

    st.divider()
    tab_intro, tab_derivation = st.tabs(["Concept", "Derivation Notes"])

    with tab_intro:
        st.markdown(data["description"])
        st.markdown("---")
        st.markdown("**Suggested viewing flow**")
        st.markdown(
            "- Watch once without pausing (get the intuition).\n"
            "- Watch again while tracking one state's value / Q-value change.\n"
            "- Finally, map the visual changes back to the equations."
        )

    with tab_derivation:
        steps = data.get("derivation_steps")
        if steps:
            render_derivation_steps(steps)
        else:
            st.info("Derivation notes are not available for this video yet.")
