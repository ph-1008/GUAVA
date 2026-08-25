#!/bin/bash
# 使用串接的动作序列渲染指定的高斯化身形象
# 用法: ./render_concatenated_with_avatar.sh <源形象目录> <输出目录名>

# 设置环境
export PYTHONPATH=/home/paohan/GUAVA:$PYTHONPATH
PYTHON=/home/paohan/anaconda3/envs/GUAVA/bin/python

# 参数
SOURCE_IMAGE=${1:-"/home/paohan/GUAVA/outputs/app/tracked_source_image/Gemini_Generated_Image_kzne4skzne4skzne"}
OUTPUT_NAME=${2:-"concatenated_render_cross"}
CONCATENATED_DATA="/home/paohan/GUAVA/outputs/concatenated_tracking_1224"

echo "=========================================="
echo "渲染串接动作序列到高斯化身"
echo "=========================================="
echo "源形象: $SOURCE_IMAGE"
echo "动作序列: $CONCATENATED_DATA"
echo "输出目录: outputs/$OUTPUT_NAME"
echo "=========================================="

# 执行渲染
cd /home/paohan/GUAVA
$PYTHON main/test.py \
  -d '0' \
  -m assets/GUAVA \
  -s outputs/$OUTPUT_NAME \
  --data_path $CONCATENATED_DATA \
  --source_data_path $SOURCE_IMAGE \
  --skip_self_act \
  --render_cross_act

echo ""
echo "✓ 渲染完成！"
echo "输出视频位置:"
echo "  outputs/$OUTPUT_NAME/render_cross_act/*/concatenated_video_*_video.mp4"
