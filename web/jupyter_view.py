import streamlit as st
import os
import nbformat
from nbconvert import HTMLExporter
import urllib.parse

def show_jupyter_module():
    st.subheader("Jupyter Notebooks")
    st.caption("View, download, or run the raw code directly in Google Colab.")

    # 1. 配置 GitHub 信息 (用于生成 Colab 链接)
    # 请确保这些信息与你的 GitHub 仓库一致
    GITHUB_USER = "sssonghuanan"
    GITHUB_REPO = "animations"
    GITHUB_BRANCH = "main" # 或者 "master"

    # 定位到存放 .ipynb 的目录
    notebook_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook")
    
    if not os.path.exists(notebook_dir):
        st.error(f"Not found notebook directory: {notebook_dir}")
        return

    # 获取所有 .ipynb 文件
    files = sorted([f for f in os.listdir(notebook_dir) if f.endswith(".ipynb")])
    
    if not files:
        st.warning("Not found in the directory .ipynb files.")
        return

    # 2. 侧边栏文件选择
    selected_file = st.sidebar.selectbox("select Notebook file", files)
    file_path = os.path.join(notebook_dir, selected_file)

    # 3. 创建操作按钮列
    col1, col2 = st.columns([1, 1])

    with col1:
        # 下载按钮
        with open(file_path, "rb") as f:
            st.download_button(
                label=f" download {selected_file}",
                data=f,
                file_name=selected_file,
                mime="application/x-ipynb+json",
                use_container_width=True
            )

    with col2:
        # 生成 Colab 链接并显示徽章按钮
        # 编码文件名以处理空格或特殊字符
        encoded_file = urllib.parse.quote(selected_file)
        colab_url = f"https://colab.research.google.com/github/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/notebook/{encoded_file}"
        
        # 使用 Markdown 渲染 Colab 标准徽章样式
        st.markdown(
            f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})'
        )

    st.divider()

    # 4. 渲染 Notebook 内容预览
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb_node = nbformat.read(f, as_version=4)
            
        html_exporter = HTMLExporter()
        html_exporter.exclude_input_prompt = True 
        (body, resources) = html_exporter.from_notebook_node(nb_node)
        
        # 显示预览
        st.components.v1.html(body, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"Unable to preview Notebook: {e}")