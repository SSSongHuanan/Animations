# Value Iteration 动画演示

使用 Manim 制作的 Value Iteration 算法教学视频，展示在 3x3 网格世界中的价值迭代过程。

## 功能特点

- 可视化 3x3 网格世界
- 实时显示价值函数的迭代更新过程
- 显示最优策略（箭头表示）
- 包含目标状态（绿色）和陷阱状态（红色）
- 自动检测收敛

## 运行方法

### 安装依赖

```bash
uv sync
```

### 渲染视频

```bash
manim -pql main.py ValueIteration
```

参数说明：
- `-p`: 渲染后自动预览
- `-ql`: 低质量（快速预览）
- `-qm`: 中等质量
- `-qh`: 高质量（最终输出）

### 完整命令示例

```bash
# 低质量预览
manim -pql main.py ValueIteration

# 高质量渲染
manim -pqh main.py ValueIteration
```

## 网格世界配置

- **普通状态**: 白色，奖励 -0.04（每步的小惩罚）
- **目标状态**: 绿色，位置 (2,2)，奖励 +1.0
- **陷阱状态**: 红色，位置 (1,1)，奖励 -1.0
- **折扣因子**: γ = 0.9

## 算法说明

Value Iteration 使用 Bellman 方程迭代更新每个状态的价值：

```
V(s) = R(s) + γ * max_a Σ P(s'|s,a) * V(s')
```

动画会展示：
1. 初始价值函数
2. 每次迭代的价值更新
3. 收敛检测
4. 最终的最优策略

## 故障排除

### 如果遇到 `InvalidDataError` 错误

这通常是由于部分视频文件损坏或缓存问题导致的。解决方法：

1. **清理缓存**（推荐）：
```bash
# 使用提供的清理脚本
python clean_cache.py

# 或手动删除 media 目录
rmdir /s /q media  # Windows
rm -rf media      # Linux/Mac
```

2. **重新渲染**：
```bash
# 清理后重新渲染
manim -pql main.py ValueIteration
```

3. **如果问题持续**，尝试使用不同的质量设置：
```bash
# 尝试低质量渲染
manim -pql main.py ValueIteration

# 或尝试预览模式（不生成完整视频）
manim --preview -ql main.py ValueIteration
```

### 其他常见问题

- **OneDrive 同步问题**：如果项目在 OneDrive 目录中，可能需要暂停同步或移动到本地目录
- **内存不足**：尝试使用低质量设置 (`-ql`) 或减少迭代次数
- **FFmpeg 问题**：确保已正确安装 FFmpeg 并添加到系统路径
