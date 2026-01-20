import os

import streamlit as st

from .common import pick_best_quality, get_manim_video, render_derivation_steps
from .algorithms import get_animation_data


def show_animation_library():
    """Render the Animation Library page."""

    st.header("Algorithm Animations")
    st.caption("Manim-based videos that visualize how values and policies evolve during learning.")

    animation_data = get_animation_data()

    st.sidebar.markdown("## Select a Video")
    keys = list(animation_data.keys())
    selected_key = st.sidebar.radio("Algorithm", keys)
    data = animation_data[selected_key]

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
