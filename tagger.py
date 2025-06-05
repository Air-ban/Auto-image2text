import os
import re
import base64
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from cut import process_image as cut_image

# ================== 工具函数 ==================

def image_to_base64(file_path):
    """将图片文件转换为base64编码"""
    with open(file_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type = "image/jpeg" if file_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return f"data:{mime_type};base64,{encoded_str}"

def validate_api_key(api_key):
    """验证 API 密钥是否有效（通过一次测试请求）"""
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
        )
        # 发起一个最小请求验证密钥有效性
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception as e:
        print(f"API 密钥验证失败: {e}")
        return False

# ================== 图片处理函数 ==================

def describe_image(image_path, api_key):
    """调用Qwen-VL模型生成图片描述"""
    try:
        base64_image = image_to_base64(image_path)

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
        )

        completion = client.chat.completions.create(
            model="qwen-vl-plus-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请使用纯英语描述图片内容，无需过长："},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ],
            timeout=30
        )

        description = completion.choices[0].message.content
        return description

    except Exception as e:
        print(f"处理 {image_path} 时出错: {str(e)}")
        return None

def process_image(image_path, api_key):
    """处理单个图片文件"""
    print(f"正在处理: {image_path}")

    description = describe_image(image_path, api_key)

    if description:
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(description)
        print(f"完成: {txt_path}")
    else:
        print(f"跳过: {image_path}")

def batch_process_images(folder_path, max_workers=3, api_key=None, target_size=None):
    """批量处理图片文件"""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]

    print(f"找到 {len(image_files)} 个图片文件，开始处理...")

    # 如果需要裁切，先创建临时文件夹并进行裁切
    if target_size:
        temp_folder = os.path.join(folder_path, "output")
        os.makedirs(temp_folder, exist_ok=True)
        print("正在进行图片裁切...")
        
        # 清空原有的文件列表，使用裁切后的图片
        cropped_files = []
        original_to_cropped = {}  # 用于跟踪原始文件和裁切文件的对应关系
        
        for image_path in tqdm(image_files, desc="裁切进度"):
            filename = os.path.basename(image_path)
            output_path = os.path.join(temp_folder, filename)
            cut_image(image_path, output_path, target_size)
            cropped_files.append(output_path)
            original_to_cropped[image_path] = output_path
        
        # 使用裁切后的文件进行处理
        files_to_process = cropped_files
    else:
        files_to_process = image_files

    print("开始生成图片描述...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, image_path, api_key) 
                  for image_path in files_to_process]

        for future in tqdm(futures, desc="处理进度"):
            # 等待每个任务完成
            future.result()

    # 如果进行了裁切，处理文件移动
    if target_size:
        # 询问是否保留裁切后的图片
        keep_cropped = input("是否保留裁切后的图片？(y/n): ").lower().strip() == 'y'
        
        for original_path, cropped_path in original_to_cropped.items():
            # 移动txt文件（从临时文件夹到原始文件夹）
            cropped_txt = os.path.splitext(cropped_path)[0] + ".txt"
            if os.path.exists(cropped_txt):
                target_txt = os.path.splitext(original_path)[0] + ".txt"
                os.rename(cropped_txt, target_txt)

            # 处理裁切后的图片
            if keep_cropped:
                # 用裁切后的图片替换原始图片
                os.rename(cropped_path, original_path)
            else:
                # 如果不保留裁切后的图片，直接删除
                if os.path.exists(cropped_path):
                    os.remove(cropped_path)

        # 删除临时文件夹
        try:
            import shutil
            shutil.rmtree(temp_folder)
        except Exception as e:
            print(f"清理临时文件时出错: {e}")

    print("全部处理完成！")

def rename_images(folder_path):
    """自动重命名图片文件为 001.jpg 格式，跳过已重命名的文件"""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    files = os.listdir(folder_path)
    to_rename = []

    for f in files:
        full_path = os.path.join(folder_path, f)
        if os.path.isdir(full_path):
            continue
        name, ext = os.path.splitext(f)
        if ext.lower() not in image_extensions:
            continue
        if re.match(r'^\d{3}$', name):
            continue
        to_rename.append(f)

    to_rename.sort()

    for idx, filename in enumerate(to_rename, 1):
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{idx:03d}{ext}"
        new_path = os.path.join(folder_path, new_name)
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_name}")

    return True

# ================== 主程序入口 ==================

if __name__ == "__main__":
    # 1. 获取并验证 API 密钥
    while True:
        api_key = input("请输入 DashScope API 密钥（以 sk- 开头）: ").strip()
        if validate_api_key(api_key):
            print("API 密钥验证成功！")
            break
        else:
            print("无效的 API 密钥，请重新输入。")

    # 2. 获取并验证文件夹路径
    while True:
        folder_path = input("请输入图片所在的文件夹路径: ").strip()
        if os.path.isdir(folder_path) and os.access(folder_path, os.R_OK):
            print("文件夹路径有效！")
            break
        else:
            print("无效的文件夹路径，请重新输入。")

    # 3. 询问是否需要裁切
    need_crop = input("是否需要对图片进行裁切？(y/n): ").lower().strip() == 'y'
    target_size = None
    
    if need_crop:
        while True:
            try:
                width = int(input("请输入目标宽度（默认1024）: ") or "1024")
                height = int(input("请输入目标高度（默认1024）: ") or "1024")
                if width > 0 and height > 0:
                    target_size = (width, height)
                    break
                else:
                    print("宽度和高度必须大于0")
            except ValueError:
                print("请输入有效的数字")

    # 4. 执行图片重命名
    rename_images(folder_path)

    # 5. 批量处理图片（包括裁切和生成描述）
    batch_process_images(folder_path, api_key=api_key, target_size=target_size)