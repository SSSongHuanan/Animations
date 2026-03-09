import streamlit as st

from .common import page_header
from . import dp, td, dqn


CHAPTERS = {
    "1. Dynamic Programming (DP)": dp,
    "2. Temporal Difference (TD)": td,
    "3. Deep Q-Networks (DQN)": dqn,
}


def show_notebook_module():
    page_header(
        "Theorey",
        "Bridge **math intuition** ↔ **implementable updates**. Choose a chapter and read it like a mini-lecture.",
    )

    # ---- Sidebar (simple + consistent) ----
    st.sidebar.markdown("## Chapters")
    topic = st.sidebar.radio("Select:", list(CHAPTERS.keys()))
    st.sidebar.divider()

    # ---- Routing ----
    CHAPTERS[topic].render()
