import streamlit as st

def show_notebook_module():
    st.title("📖 Theoretical Notebooks")
    st.markdown("""
    Welcome to the **Digital Notebook**. Here we bridge the gap between abstract mathematics and executable code.
    Select a topic from the sidebar to dive into the algorithmic mechanics.
    """)

    # 1. Sidebar for Blog Navigation
    topic = st.sidebar.radio(
        "Select Chapter:",
        [
            "1. Dynamic Programming (DP)",
            "2. Temporal Difference (TD)", 
            "3. Deep Q-Networks (DQN)"
        ]
    )

    st.divider()

    # 2. Content Routing
    if topic == "1. Dynamic Programming (DP)":
        render_dp_article()
    elif topic == "2. Temporal Difference (TD)":
        render_td_article()
    elif topic == "3. Deep Q-Networks (DQN)":
        render_dqn_article()

# ==========================================
# Article 1: Dynamic Programming
# ==========================================
def render_dp_article():
    st.header("1. Dynamic Programming: Planning in a Known World")
    st.caption("Keywords: Model-Based, Bellman Equation, Bootstrap")

    st.markdown("""
    ### 1.1 The Premise
    Dynamic Programming (DP) methods, such as **Policy Iteration** and **Value Iteration**, assume that the agent has a perfect map of the world. In RL terms, this means we know the **Transition Probability** $P(s'|s,a)$ and the **Reward Function** $R(s,a)$.
    
    Because we know the rules of the game, we don't need to explore blindly. We can simply "plan" by solving systems of equations.

    ### 1.2 The Bellman Optimality Equation
    The core of DP is the recursive relationship between the value of a state and the value of its successors. The optimal value function $V^*(s)$ satisfies:
    """)

    st.latex(r"V^*(s) = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma V^*(s')]")

    st.markdown("""
    ### 1.3 Policy Iteration vs. Value Iteration
    While both algorithms converge to the same optimal policy, they take different paths:

    * **Policy Iteration** separates the process into two distinct, alternating phases:
        1.  **Evaluation:** Calculate $V_\pi(s)$ for the current policy (often computationally expensive).
        2.  **Improvement:** Update $\pi$ to be greedy with respect to $V_\pi$.
    
    * **Value Iteration** truncates the evaluation step. It iterates on the value function directly using the "max" operator. It effectively combines evaluation and improvement into a single, aggressive update sweep.
    
    ### 1.4 Pseudocode (Value Iteration)
    """)

    st.code("""
# Initialize V(s) arbitrarily
while delta > theta:
    delta = 0
    for s in States:
        v = V[s]
        # The "Max" operator acts as the improvement
        V[s] = max([sum([p * (r + gamma * V[s_]) 
                   for p, s_, r, _ in env.P[s][a]]) 
                   for a in Actions])
        delta = max(delta, abs(v - V[s]))
    """, language="python")

# ==========================================
# Article 2: Temporal Difference
# ==========================================
def render_td_article():
    st.header("2. Temporal Difference Learning: Learning from Experience")
    st.caption("Keywords: Model-Free, Q-Learning, SARSA, Bootstrapping")

    st.markdown("""
    ### 2.1 Moving Beyond the Model
    In most real-world scenarios (like driving a car or trading stocks), we do **not** know the transition dynamics $P(s'|s,a)$. We cannot calculate expectations directly. Instead, we must sample the environment.
    
    **Temporal Difference (TD)** learning combines two ideas:
    1.  **Monte Carlo:** Learn from raw experience (samples).
    2.  **DP:** Update estimates based on other learned estimates (bootstrapping).

    ### 2.2 SARSA (On-Policy)
    SARSA is "safe". It updates the Q-value based on the action the agent *actually* took.
    
    $$Q(S, A) \\leftarrow Q(S, A) + \\alpha [R + \\gamma Q(S', A') - Q(S, A)]$$
    
    If the agent explores (takes a random bad action $A'$), SARSA will penalize the previous state $S$. This leads to safer, more conservative paths (e.g., staying far away from the cliff).

    ### 2.3 Q-Learning (Off-Policy)
    Q-Learning is "optimistic". It updates the Q-value assuming the agent will take the *best* possible action next, regardless of what it actually does.
    
    $$Q(S, A) \\leftarrow Q(S, A) + \\alpha [R + \\gamma \\max_{a} Q(S', a) - Q(S, A)]$$
    
    Even if the agent falls off a cliff due to exploration ($\epsilon$-greedy), Q-learning considers that a "random accident" and continues to learn the optimal path right next to the edge.

    ### 2.4 Comparison Snippet
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SARSA Update**")
        st.code("""
next_action = choose_action(next_state)
target = reward + gamma * Q[next_state][next_action]
error = target - Q[state][action]
Q[state][action] += alpha * error
        """, language="python")
    with col2:
        st.markdown("**Q-Learning Update**")
        st.code("""
# No next_action needed for target
target = reward + gamma * np.max(Q[next_state])
error = target - Q[state][action]
Q[state][action] += alpha * error
        """, language="python")

# ==========================================
# Article 3: DQN
# ==========================================
def render_dqn_article():
    st.header("3. Deep Q-Networks (DQN): Scaling Up")
    st.caption("Keywords: Neural Networks, Function Approximation, Replay Buffer")

    st.markdown("""
    ### 3.1 The Curse of Dimensionality
    Tabular methods (creating a matrix for $Q(s,a)$) work for small grid worlds. But what if the state is an image (Atari games)? A $84 \\times 84$ image has $256^{84 \\times 84}$ possible states. A table is impossible.
    
    **Solution:** Use a Neural Network to approximate the function: $Q(s, a; \\theta) \\approx Q^*(s, a)$.

    ### 3.2 Stabilizing the Unstable
    Naively connecting a Deep Network to RL is unstable because RL data is not i.i.d. (independent and identically distributed). DQN introduced two key innovations:

    #### A. Experience Replay
    Instead of learning from the current step immediately, we store transitions $(s, a, r, s')$ in a massive buffer. We then sample a random **batch** to train the network. This breaks the temporal correlation between samples.

    #### B. Target Network
    In standard Q-learning, the target changes every step. This is like a dog chasing its own tail.
    DQN freezes the "Target Network" parameters $\\theta^-$ for fixed intervals (e.g., every 1000 steps), creating a stable target for the "Policy Network" to learn towards.

    ### 3.3 The Loss Function
    The network minimizes the Mean Squared Error (MSE) between the prediction and the target:
    """)

    st.latex(r"L(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]")
    
    st.info("The gradient descent update allows the agent to generalize: learning about one state helps it understand similar states.")