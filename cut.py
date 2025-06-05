import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def detect_faces(image):
    """使用OpenCV的Haar级联检测人脸"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) > 0:
        return max(faces, key=lambda x: x[2] * x[3])  # 返回最大人脸
    return None

def detect_saliency(image):
    """使用显著性检测寻找视觉焦点"""
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(image)
    if not success:
        return None
    thresh_map = cv2.threshold(saliency_map.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return cv2.boundingRect(max(contours, key=cv2.contourArea))
    return None

def get_focus_box(image):
    """尝试检测人脸或显著区域作为焦点"""
    face_box = detect_faces(image)
    if face_box is not None:
        return face_box
    return detect_saliency(image)

def calculate_crop_region(focus_box, image_size, target_size):
    """根据焦点计算裁切区域，确保不越界"""
    img_w, img_h = image_size
    target_w, target_h = target_size
    fx, fy, fw, fh = focus_box
    
    # 计算焦点中心
    center_x = fx + fw / 2
    center_y = fy + fh / 2

    # 计算裁剪区域
    start_x = int(center_x - target_w / 2)
    start_y = int(center_y - target_h / 2)

    # 确保不越界
    start_x = max(0, min(start_x, img_w - target_w))
    start_y = max(0, min(start_y, img_h - target_h))

    # 确保裁剪区域不超过图片大小
    crop_w = min(target_w, img_w)
    crop_h = min(target_h, img_h)

    return (start_x, start_y, crop_w, crop_h)

def process_image(input_path, output_path, target_size):
    """处理单张图片：焦点检测、裁切、缩放、保存"""
    image = None
    try:
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图片: {input_path}")

        img_h, img_w = image.shape[:2]
        target_w, target_h = target_size

        # 检查输入图片尺寸是否足够
        if img_w < target_w or img_h < target_h:
            print(f"警告: 图片 {input_path} 尺寸({img_w}x{img_h})小于目标尺寸({target_w}x{target_h})")
            # 如果原图太小，直接调整到目标大小
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, resized)
            return

        focus_box = get_focus_box(image)

        if focus_box is None:
            # 如果没有检测到焦点，使用中心裁剪
            center_x = img_w / 2
            center_y = img_h / 2
            start_x = int(center_x - target_size[0] / 2)
            start_y = int(center_y - target_size[1] / 2)
            start_x = max(0, min(start_x, img_w - target_size[0]))
            start_y = max(0, min(start_y, img_h - target_size[1]))
            crop_region = (start_x, start_y, target_size[0], target_size[1])
        else:
            crop_region = calculate_crop_region(focus_box, (img_w, img_h), target_size)

        x, y, w, h = crop_region
        cropped = image[y:y+h, x:x+w]
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, resized)

    except Exception as e:
        print(f"[错误] 处理失败: {input_path} - {str(e)}")
    finally:
        # 释放资源
        if image is not None:
            del image

def bulk_process(input_dir, output_dir, target_size, recursive):
    """批量处理文件夹中的图片"""
    supported_extensions = ('.jpg', '.jpeg', '.png')
    file_paths = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                input_path = os.path.join(root, file)
                file_paths.append(input_path)
        if not recursive:
            break  # 不进入子文件夹

    # 使用 tqdm 显示进度条
    for input_path in tqdm(file_paths, desc="处理图片", unit="张"):
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        output_path = os.path.splitext(output_path)[0] + os.path.splitext(input_path)[1]
        process_image(input_path, output_path, target_size)

def main():
    parser = argparse.ArgumentParser(description='自动裁切文件夹中的图片到指定分辨率')
    parser.add_argument('--input', required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', required=True, help='输出图片文件夹路径')
    parser.add_argument('--width', type=int, default=1024, help='目标宽度')
    parser.add_argument('--height', type=int, default=1024, help='目标高度')
    parser.add_argument('--recursive', action='store_true', help='递归处理子文件夹')

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"输入路径不是有效文件夹: {args.input}")
        return

    target_size = (args.width, args.height)
    bulk_process(args.input, args.output, target_size, args.recursive)
    print("✅ 图片处理完成！")

if __name__ == "__main__":
    main()