import streamlit as st
from .cliff_walking import run as run_cliff
from .frozen_lake import run as run_frozen

def run_experiments_module():
    st.title('Interactive RL Laboratory')
    st.sidebar.markdown('## Select Experiment')

    EXP_MAP = {
        'Cliff Walking': run_cliff,
        'Frozen Lake': run_frozen,
    }

    default = st.session_state.get('exp_type', 'Cliff Walking')
    options = list(EXP_MAP.keys())
    if default not in options:
        default = options[0]

    exp_type = st.sidebar.radio('Environment:', options, index=options.index(default), key='exp_type')
    st.sidebar.divider()

    EXP_MAP[exp_type]()