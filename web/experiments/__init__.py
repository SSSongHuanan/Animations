import streamlit as st
from .cliff_walking import run as run_cliff
from .frozen_lake import run as run_frozen

def run_experiments_module():
    st.title("🧪 Interactive RL Laboratory")
    st.sidebar.markdown("## 🧪 Select Experiment")
    
    # 以后你想加新实验，只需要新建文件，然后在这里注册即可
    EXP_MAP = {
        "Cliff Walking": run_cliff,
        "Frozen Lake": run_frozen
    }
    
    exp_type = st.sidebar.radio("Environment:", list(EXP_MAP.keys()))
    st.sidebar.divider()
    
    if exp_type in EXP_MAP:
        EXP_MAP[exp_type]()