# Auto-image2text

## 项目说明
本工具实现自动化图像分割与文本标签生成，包含以下功能：
- `cut.py`：基于OpenCV的图像切割模块
- `tagger.py`：使用CLIP模型生成图像描述

## 使用方法
1. 安装依赖：`pip install -r requirements.txt`
2. 运行图像切割：`python cut.py --input 图像路径`
3. 生成文本标签：`python tagger.py --image 图像路径`