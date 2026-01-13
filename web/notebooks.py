import streamlit as st

# =====================================================
# Theoretical Notebooks (Optimized UI)
# - Consistent layout with your other modules
# - Cleaner typography + right-side "Key ideas" cards
# - Tabs per chapter: Concept / Math / Pseudocode / Checkpoint
# =====================================================

def _page_header(title: str, subtitle: str):
    st.header(title)
    st.caption(subtitle)
    st.divider()


def _right_card(title: str, bullets=None, body=None):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if bullets:
            st.markdown("\n".join([f"- {b}" for b in bullets]))
        if body:
            st.markdown(body)


def _checkpoint(items):
    st.markdown("#### Self-check")
    for it in items:
        st.checkbox(it, value=False)


def show_notebook_module():
    _page_header(
        "Theoretical Notebooks",
        "Bridge **math intuition** ↔ **implementable updates**. Choose a chapter and read it like a mini-lecture.",
    )

    # ---- Sidebar (simple + consistent) ----
    st.sidebar.markdown("## Chapters")
    topic = st.sidebar.radio(
        "Select:",
        [
            "1. Dynamic Programming (DP)",
            "2. Temporal Difference (TD)",
            "3. Deep Q-Networks (DQN)",
        ],
    )
    st.sidebar.divider()

    # ✅ 修复点：多行 markdown 用三引号，避免 Python 字符串跨行语法错误
    with st.sidebar.expander("How to use this page", expanded=False):
        st.markdown(
            """
- Read **Concept** first for intuition.
- Then check **Math** for the exact equations.
- Use **Pseudocode** to connect to code.
- Finish with **Checkpoint** to test understanding.
            """
        )

    # ---- Routing ----
    if topic.startswith("1."):
        render_dp_article()
    elif topic.startswith("2."):
        render_td_article()
    else:
        render_dqn_article()


# =====================================================
# Chapter 1: Dynamic Programming
# =====================================================
def render_dp_article():
    st.subheader("1. Dynamic Programming (DP)")
    st.caption("Model-based planning with known dynamics. Keywords: Bellman equation, backup, bootstrapping.")

    left, right = st.columns([1.6, 1.0], gap="large")

    with right:
        _right_card(
            "Key ideas",
            bullets=[
                "DP assumes you know the environment model: p(s', r | s, a).",
                "You can plan without exploration by repeatedly applying Bellman backups.",
                "Policy Iteration: evaluate then improve. Value Iteration: combine them.",
            ],
        )
        _right_card(
            "When to use",
            bullets=[
                "Small/medium discrete MDPs with known dynamics.",
                "As a gold-standard baseline for RL environments (ground-truth optimal).",
            ],
        )

    with left:
        tab_concept, tab_math, tab_code, tab_check = st.tabs(
            ["Concept", "Math", "Pseudocode", "Checkpoint"]
        )

        with tab_concept:
            st.markdown(
                """
### 1.1 Premise
Dynamic Programming methods assume the agent has a *perfect map* of the world:
- Transition model: **p(s′, r | s, a)**
- Reward function is included in that model (or available separately)

Because the rules are known, we don’t need to explore blindly—we can **compute** expected returns.
                """
            )
            st.info("DP is the planning baseline: it solves the MDP directly via repeated backups.")

        with tab_math:
            st.markdown("### 1.2 Bellman Optimality Backup (state value)")
            st.latex(r"V^*(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\,[r+\gamma V^*(s')]")

            st.markdown("### 1.3 Policy Iteration vs. Value Iteration (high level)")
            st.markdown(
                "- **Policy Iteration**: alternate *Policy Evaluation* and *Policy Improvement*.\n"
                "- **Value Iteration**: repeatedly apply the optimality backup; derive policy from V."
            )

        with tab_code:
            st.markdown("### 1.4 Value Iteration (pseudocode)")
            st.code(
                """# Initialize V(s) arbitrarily
while delta > theta:
    delta = 0
    for s in States:
        v = V[s]
        V[s] = max_a sum_{s',r} p(s',r|s,a) * (r + gamma * V[s'])
        delta = max(delta, abs(v - V[s]))
""",
                language="python",
            )

        with tab_check:
            _checkpoint(
                [
                    "I can explain what 'model-based' means in DP.",
                    "I can read the Bellman optimality equation and identify each term.",
                    "I know the difference between Policy Iteration and Value Iteration.",
                ]
            )


# =====================================================
# Chapter 2: Temporal Difference
# =====================================================
def render_td_article():
    st.subheader("2. Temporal Difference (TD) Learning")
    st.caption("Model-free learning from experience. Keywords: TD error, on-policy, off-policy.")

    left, right = st.columns([1.6, 1.0], gap="large")

    with right:
        _right_card(
            "Key ideas",
            bullets=[
                "TD learns from samples (no model needed).",
                "Bootstrapping: update uses current estimates as targets.",
                "SARSA is on-policy (uses actual next action). Q-learning is off-policy (uses max).",
            ],
        )
        _right_card(
            "Mental model",
            bullets=[
                "SARSA learns a policy that matches its exploration behavior (often safer).",
                "Q-learning learns the greedy optimal policy, even while exploring (often riskier).",
            ],
        )

    with left:
        tab_concept, tab_math, tab_code, tab_check = st.tabs(
            ["Concept", "Math", "Code Snippets", "Checkpoint"]
        )

        with tab_concept:
            st.markdown(
                """
### 2.1 Why TD?
In most real-world problems we **don't** know the transition model. TD methods learn from interaction.

TD combines:
1) **Monte Carlo**: learn from experience samples  
2) **DP**: bootstrap from existing value estimates
                """
            )

        with tab_math:
            st.markdown("### 2.2 SARSA (On-policy)")
            st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma Q(s',a')-Q(s,a)]")

            st.markdown("### 2.3 Q-learning (Off-policy)")
            st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]")

            st.info("Key difference: SARSA targets Q(s′,a′) from the behavior policy; Q-learning targets max over actions.")

        with tab_code:
            st.markdown("**SARSA update**")
            st.code(
                """next_action = choose_action(next_state)  # ε-greedy
        target = reward + gamma * Q[next_state][next_action]
        Q[state][action] += alpha * (target - Q[state][action])
        """,
                language="python",
            )

            st.markdown("**Q-learning update**")
            st.code(
                """target = reward + gamma * max(Q[next_state])
        Q[state][action] += alpha * (target - Q[state][action])
        """,
                language="python",
            )

            st.info("Difference: SARSA uses Q(s′,a′) (on-policy), while Q-learning uses maxₐ′Q(s′,a′) (off-policy).")


        with tab_check:
            _checkpoint(
                [
                    "I can define TD error in words (prediction error).",
                    "I can explain on-policy vs off-policy using SARSA vs Q-learning.",
                    "I can predict which one is safer in CliffWalking and why.",
                ]
            )


# =====================================================
# Chapter 3: DQN
# =====================================================
def render_dqn_article():
    st.subheader("3. Deep Q-Networks (DQN)")
    st.caption("Scaling Q-learning with function approximation. Keywords: replay buffer, target network.")

    left, right = st.columns([1.6, 1.0], gap="large")

    with right:
        _right_card(
            "Key ideas",
            bullets=[
                "Replace Q-table with a neural network Q(s,a; θ).",
                "Replay buffer makes training data closer to i.i.d.",
                "Target network θ⁻ stabilizes the bootstrapped target.",
            ],
        )
        _right_card(
            "Common failure modes",
            bullets=[
                "Divergence from moving targets (no target network).",
                "Correlated samples (no replay).",
                "Overestimation bias (often improved with Double DQN).",
            ],
        )

    with left:
        tab_concept, tab_math, tab_code, tab_check = st.tabs(
            ["Concept", "Math", "Training Loop", "Checkpoint"]
        )

        with tab_concept:
            st.markdown(
                """
### 3.1 Curse of dimensionality
Tabular methods work for small discrete state spaces. With images or continuous states, tables are impossible.

DQN approximates the action-value function with a neural network:  
**Q(s,a; θ) ≈ Q*(s,a)**

### 3.2 Two stabilizers
- **Experience replay**: store transitions and sample random minibatches.
- **Target network**: compute targets using a slowly-updated copy of parameters.
                """
            )

        with tab_math:
            st.markdown("### 3.3 TD target and loss")
            st.latex(r"y = r + \gamma\max_{a'}Q(s',a';\theta^-)")
            st.latex(r"L(\theta)=\mathbb{E}\big[(y - Q(s,a;\theta))^2\big]")
            st.info("Intuition: fit Q(s,a;θ) to a relatively stable target y computed using θ⁻.")

        with tab_code:
            st.code(
                """# Pseudocode (high level)
replay = ReplayBuffer()
for step in range(num_steps):
    a = epsilon_greedy(Q, s)
    s2, r, done = env.step(a)
    replay.add(s, a, r, s2, done)

    batch = replay.sample(B)
    y = r + gamma * (1-done) * max_a' Q_target(s2, a')
    loss = mse(Q(s,a), y)
    optimizer.step(loss)

    if step % target_update == 0:
        Q_target.load_state_dict(Q.state_dict())
""",
                language="python",
            )

        with tab_check:
            _checkpoint(
                [
                    "I can explain why a Q-table fails for image inputs.",
                    "I can explain why replay helps.",
                    "I can explain why a target network helps.",
                ]
            )
