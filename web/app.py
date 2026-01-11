import streamlit as st
import sys
import os

# --- 关键修复 1: 路径设置 ---
# 将项目根目录添加到 python path，这样如果有 shared utils 在根目录也能被找到
# 同时也确保 streamlit 能正确找到 media 文件夹
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../animation/web
root_dir = os.path.dirname(current_dir)                  # .../animation
sys.path.append(root_dir)

# --- 关键修复 2: 修正导入语句 ---
# 错误写法: from web.animations import ...
# 正确写法: 直接 import 同级文件，或者从子文件夹 import
from animations import show_animation_library
from experiments import run_experiments_module
from notebooks import show_notebook_module

# 1. Page Config
st.set_page_config(page_title="RL Education Platform", layout="wide", page_icon="🎓")

# 2. Sidebar Navigation
st.sidebar.header("🧭 Platform Navigation")

app_mode = st.sidebar.radio(
    "Choose Module:", 
    [
        "🎥 Animation Library", 
        "📖 Theory Notebooks",
        "🧪 RL Laboratory"
    ]
)

st.sidebar.divider()
st.sidebar.info("Developed by Song Huanan | BUPT & QMUL Joint Program")

# 3. Router
if app_mode == "📖 Theory Notebooks":
    show_notebook_module()

elif app_mode == "🎥 Animation Library":
    show_animation_library()

elif app_mode == "🧪 RL Laboratory":
    run_experiments_module()