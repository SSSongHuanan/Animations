import os

def get_manim_video_path(scene_name, quality="1080p60", video_name=None):
    """
    根据 Manim 的输出规则自动寻找视频文件路径。
    
    :param scene_name: 你在 Manim 代码类定义的名字 (例如 class DQNDemo(Scene): 中的 DQNDemo)
    :param quality: 默认是 1080p60
    :param video_name: 视频文件名，通常和 scene_name 一样，如果未指定则默认用 scene_name
    """
    if video_name is None:
        video_name = scene_name
        
    base_path = os.path.join("media", "videos", scene_name, quality, f"{video_name}.mp4")
    
    if os.path.exists(base_path):
        return base_path
    else:
        print(f"Warning: Video not found at {base_path}")
        return None