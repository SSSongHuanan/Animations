import streamlit as st
import sys
import os
from home import show_home
from animations import show_animation_library
from experiments import run_experiments_module
from notebooks import show_notebook_module

# --- Path setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../animation/web
root_dir = os.path.dirname(current_dir)                   # .../animation
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
st.set_page_config(page_title='RL Education Platform', layout='wide', page_icon='')

st.sidebar.header('Platform Navigation')

if 'app_mode' not in st.session_state:
    st.session_state['app_mode'] = 'Home'

app_mode = st.sidebar.radio(
    label='',
    options=['Home', 'Animation Library', 'Theory Notebooks', 'RL Laboratory'],
    key='app_mode',
    label_visibility='collapsed',
)

st.sidebar.divider()


if app_mode == 'Home':
    show_home()
elif app_mode == 'Animation Library':
    show_animation_library()
elif app_mode == 'Theory Notebooks':
    show_notebook_module()
elif app_mode == 'RL Laboratory':
    run_experiments_module()

st.sidebar.info('Developed by Song Huanan')