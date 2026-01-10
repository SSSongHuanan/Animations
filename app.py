import streamlit as st
# 导入我们刚刚写好的两个模块
from web_app.animations import show_animation_library
from web_app.experiments import run_cliff_walking_experiment

# 1. 页面配置
st.set_page_config(page_title="RL Education Platform", layout="wide", page_icon="🎓")

# 2. 侧边栏导航
st.sidebar.header("🧭 Platform Navigation")

app_mode = st.sidebar.radio(
    "Choose Module:", 
    [
        "🎥 Animation Library", 
        "🧪 RL Laboratory"
    ]
)

st.sidebar.divider()
st.sidebar.info("Developed by Song Huanan | BUPT & QMUL")

# 3. 路由逻辑 (Router)
if app_mode == "🎥 Animation Library":
    # 调用 animations.py 里的函数
    show_animation_library()

elif app_mode == "🧪 RL Laboratory":
    # 调用 experiments.py 里的函数
    run_cliff_walking_experiment()