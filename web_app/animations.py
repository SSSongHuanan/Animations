import streamlit as st
import os

# 把路径查找逻辑封装在这里，或者从 utils 引入
def get_manim_video(scene_folder, video_name=None, quality="1080p60"):
    if video_name is None:
        video_name = scene_folder
    # 注意：这里的路径是相对于主运行文件(app.py)的
    path = os.path.join("media", "videos", scene_folder, quality, f"{video_name}.mp4")
    if os.path.exists(path):
        return path
    return None

def show_animation_library():
    st.title("🎥 Algorithm Animation Library")
    st.markdown("""
    **Concept Learning:** Watch these programmatic animations to understand the 
    mathematical "ripple effects" and temporal dynamics of RL algorithms.
    """)

    # 你的视频字典
    video_dict = {
        "Deep Q-Network (DQN)": ("DQN", "DQNDemo"),
        "Policy Iteration": ("Policy_iteration", "PolicyIteration"), 
        "Q-Learning": ("QLearning", "QLearningDemo"),
        "SARSA": ("SARSA", "SARSADemo")
    }

    selected_video = st.selectbox("Select an Algorithm to Watch:", list(video_dict.keys()))

    folder, filename = video_dict[selected_video]
    video_path = get_manim_video(folder, filename)

    if video_path:
        st.success(f"Now Playing: {selected_video}")
        st.video(video_path)
    else:
        st.error(f"❌ Video file not found.")
        st.code(f"Path searched: media/videos/{folder}/1080p60/{filename}.mp4")