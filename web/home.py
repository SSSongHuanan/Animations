import streamlit as st

def _set_mode(mode: str):
    st.session_state['app_mode'] = mode

def show_home():
    st.header('🎓 RL Education Platform')
    st.caption('Programmatic animations + interactive experiments for Reinforcement Learning.')

    st.markdown(
        'Reinforcement Learning is **dynamic**: values and policies change over time, but most learning materials are **static**.\n'
        'This site bridges that gap by combining **Manim-based visual explanations** with **Gymnasium-based hands-on experiments**.'
    )

    st.divider()

    c1, c2, c3 = st.columns(3, gap='large')

    with c1:
        with st.container(border=True):
            st.markdown('### 🎥 Animation Library')
            st.markdown('- Watch *programmatic* Manim videos\n- See **value propagation** / **policy updates**\n- Follow the “concrete → abstract” narrative')
            st.button('Open Animations', use_container_width=True, on_click=_set_mode, args=('🎥 Animation Library',))

    with c2:
        with st.container(border=True):
            st.markdown('### 📖 Theory Notebooks')
            st.markdown('- Mini-lectures (Concept → Math → Pseudocode)\n- Step-wise derivations\n- Self-check questions')
            st.button('Open Notebooks', use_container_width=True, on_click=_set_mode, args=('📖 Theory Notebooks',))

    with c3:
        with st.container(border=True):
            st.markdown('### 🧪 RL Laboratory')
            st.markdown('- Run algorithms in classic Gym environments\n- Tune **α, γ, ε** in real time\n- Observe reward curves & learned policies')
            st.button('Open Laboratory', use_container_width=True, on_click=_set_mode, args=('🧪 RL Laboratory',))

    st.divider()

    left, right = st.columns([1.6, 1.0], gap='large')
    with left:
        st.subheader('🧭 Recommended learning path')
        st.markdown('1) **Watch** an animation to build intuition (What changes over time?)\n'
                   '2) **Read** the notebook to connect visuals ↔ equations ↔ pseudocode\n'
                   '3) **Experiment** in the lab to see how hyperparameters change behavior')
        st.info('Tip: Try **CliffWalking** with low γ vs high γ to feel “short-sighted vs far-sighted” behavior.')
    with right:
        with st.container(border=True):
            st.markdown('#### Project goals')
            st.markdown('- Reduce the “intuition gap” in RL learning\n- Link **visual models** with **verbal models** (math/code)\n- Enable interactive parameter exploration')
            st.caption('Built with Python • Manim • Gymnasium • Streamlit')
