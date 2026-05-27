#!/bin/bash
# 手动设置 pytorch-i3d 的脚本

HUB_DIR="/root/.cache/torch/hub"
EXPECTED_NAME="piergiaj_pytorch-i3d_05783d1"

echo "=========================================="
echo "设置手动下载的 pytorch-i3d"
echo "=========================================="
echo ""
echo "Torch Hub 目录: $HUB_DIR"
echo "期望的目录名: $EXPECTED_NAME"
echo ""

# 检查用户下载的目录在哪里
echo "请告诉我你下载的目录的完整路径："
echo "例如: /root/Downloads/piergiaj-pytorch-i3d-05783d1"
echo ""
read -p "输入目录路径: " DOWNLOADED_DIR

if [ ! -d "$DOWNLOADED_DIR" ]; then
    echo "❌ 错误: 目录不存在: $DOWNLOADED_DIR"
    exit 1
fi

# 创建目标目录
TARGET_DIR="$HUB_DIR/$EXPECTED_NAME"

if [ -d "$TARGET_DIR" ]; then
    echo "⚠️  警告: 目标目录已存在: $TARGET_DIR"
    read -p "是否覆盖? (y/n): " OVERWRITE
    if [ "$OVERWRITE" != "y" ]; then
        echo "取消操作"
        exit 0
    fi
    rm -rf "$TARGET_DIR"
fi

# 复制目录
echo ""
echo "正在复制目录..."
cp -r "$DOWNLOADED_DIR" "$TARGET_DIR"

# 检查是否成功
if [ -d "$TARGET_DIR" ]; then
    echo "✅ 成功! 目录已复制到: $TARGET_DIR"
    echo ""
    echo "现在可以运行: python eval_fvd.py"
else
    echo "❌ 错误: 复制失败"
    exit 1
fi
