"""
可视化COCO格式的真实标注框 (Ground Truth)
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

# ChestX_Det类别定义
CHEST_CATEGORIES = {
    1: 'Atelectasis',
    2: 'Calcification', 
    3: 'Cardiomegaly',
    4: 'Consolidation',
    5: 'DiffuseNodule',
    6: 'Effusion',
    7: 'Emphysema',
    8: 'Fibrosis',
    9: 'Fracture',
    10: 'Mass',
    11: 'Nodule',
    12: 'Pleural Thickening',
    13: 'Pneumothorax'
}

# 颜色列表
COLOR_LIST = [
    (255, 0, 0),         # 红色
    (0, 255, 0),         # 绿色
    (0, 0, 255),         # 蓝色
    (255, 165, 0),       # 橙色
    (255, 255, 0),       # 黄色
    (0, 255, 255),       # 青色
    (255, 0, 255),       # 品红
    (128, 0, 128),       # 紫色
    (255, 192, 203),     # 粉色
    (0, 128, 0),         # 深绿色
    (0, 0, 128),         # 深蓝色
    (128, 128, 0),       # 橄榄色
    (0, 128, 128),       # 蓝绿色
]


def get_color_by_class(class_id):
    """根据类别ID返回颜色"""
    return COLOR_LIST[(class_id - 1) % len(COLOR_LIST)]


def get_font(size):
    """获取字体"""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()


def draw_ground_truth(image_path, annotations, categories_dict, 
                      font_size_factor=0.05, box_thickness_factor=0.005,
                      show_area=False):
    """
    在图像上绘制真实标注框
    
    Args:
        image_path: 图像路径
        annotations: 该图像的所有标注列表
        categories_dict: 类别ID到名称的映射
        font_size_factor: 字体大小因子
        box_thickness_factor: 框粗细因子
        show_area: 是否显示标注框面积
    """
    im = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(im)
    w, h = im.size
    
    for ann in annotations:
        # COCO格式: [x, y, width, height]
        x, y, width, height = ann['bbox']
        category_id = ann['category_id']
        
        # 转换为 [x1, y1, x2, y2]
        box = [x, y, x + width, y + height]
        
        # 获取颜色和类别名称
        color = get_color_by_class(category_id)
        category_name = categories_dict.get(category_id, f"ID_{category_id}")
        
        # 计算字体大小
        box_width = width
        box_height = height
        font_size = max(int(min(box_width, box_height) * font_size_factor), 12)
        font = get_font(font_size)
        
        # 绘制矩形框
        box_thickness = max(int(min(w, h) * box_thickness_factor), 2)
        draw.rectangle(box, outline=color, width=box_thickness)
        
        # 准备文本
        if show_area:
            area = ann.get('area', width * height)
            text = f"{category_name} (A:{int(area)})"
        else:
            text = f"{category_name}"
        
        # 获取文本边界
        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = box[0]
        text_y = box[1] - text_height - 5
        
        # 确保文本在图像内
        if text_x + text_width > w:
            text_x = w - text_width
        if text_y < 0:
            text_y = box[1] + 5
        
        # 绘制文本背景
        background_box = [
            text_x - 2,
            text_y - 2,
            text_x + text_width + 2,
            text_y + text_height + 2
        ]
        draw.rectangle(background_box, fill=color)
        
        # 绘制文本（使用白色或黑色，根据背景色自动选择）
        text_color = (255, 255, 255) if sum(color) < 382 else (0, 0, 0)
        draw.text((text_x, text_y), text=text, fill=text_color, font=font)
    
    return im


def load_coco_annotations(annotation_file):
    """加载COCO格式的标注文件"""
    print(f"加载标注文件: {annotation_file}")
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # 创建类别字典
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 创建图像ID到标注的映射
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # 创建图像ID到图像信息的映射
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    return categories_dict, img_id_to_anns, img_id_to_info


def visualize_single_image(image_path, annotation_file, output_path, 
                           show_area=False):
    """可视化单张图片的真实标注"""
    categories_dict, img_id_to_anns, img_id_to_info = load_coco_annotations(annotation_file)
    
    # 从图像路径获取文件名
    image_name = os.path.basename(image_path)
    
    # 查找对应的图像ID和标注
    img_id = None
    for id_, info in img_id_to_info.items():
        if info['file_name'] == image_name:
            img_id = id_
            break
    
    if img_id is None:
        print(f"错误: 在标注文件中找不到图像 {image_name}")
        return
    
    annotations = img_id_to_anns.get(img_id, [])
    
    if not annotations:
        print(f"警告: 图像 {image_name} 没有标注")
    else:
        print(f"找到 {len(annotations)} 个标注框")
    
    # 绘制标注
    result_image = draw_ground_truth(image_path, annotations, categories_dict, 
                                     show_area=show_area)
    
    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"gt_{image_name}")
    result_image.save(output_file)
    print(f"✅ 保存到: {output_file}")


def visualize_dataset(image_dir, annotation_file, output_path, 
                      max_images=None, show_area=False):
    """可视化整个数据集的真实标注"""
    categories_dict, img_id_to_anns, img_id_to_info = load_coco_annotations(annotation_file)
    
    os.makedirs(output_path, exist_ok=True)
    
    # 获取所有图像
    images_to_process = list(img_id_to_info.items())
    if max_images:
        images_to_process = images_to_process[:max_images]
    
    print(f"\n处理 {len(images_to_process)} 张图像...")
    
    for img_id, img_info in tqdm(images_to_process):
        image_path = os.path.join(image_dir, img_info['file_name'])
        
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在 {image_path}")
            continue
        
        annotations = img_id_to_anns.get(img_id, [])
        
        # 绘制标注
        result_image = draw_ground_truth(image_path, annotations, categories_dict,
                                        show_area=show_area)
        
        # 保存结果
        output_file = os.path.join(output_path, f"gt_{img_info['file_name']}")
        result_image.save(output_file)
    
    print(f"\n✅ 完成！结果保存在: {output_path}")
    print(f"   共处理 {len(images_to_process)} 张图像")


def print_statistics(annotation_file):
    """打印标注统计信息"""
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    print("\n" + "="*60)
    print("标注统计信息")
    print("="*60)
    
    print(f"\n总图像数: {len(coco_data['images'])}")
    print(f"总标注数: {len(coco_data['annotations'])}")
    print(f"类别数: {len(coco_data['categories'])}")
    
    print("\n类别列表:")
    for cat in coco_data['categories']:
        print(f"  {cat['id']:2d}. {cat['name']}")
    
    # 统计每个类别的标注数
    cat_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
    
    print("\n每个类别的标注数:")
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    for cat_id, count in sorted(cat_counts.items()):
        cat_name = categories_dict.get(cat_id, f"ID_{cat_id}")
        print(f"  {cat_name:25s}: {count:4d} 个标注")
    
    # 统计有标注的图像数
    img_with_anns = len(set(ann['image_id'] for ann in coco_data['annotations']))
    print(f"\n有标注的图像数: {img_with_anns}")
    print(f"无标注的图像数: {len(coco_data['images']) - img_with_anns}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='可视化COCO格式的真实标注框')
    parser.add_argument('-a', '--annotation', type=str, required=True,
                       help='COCO标注文件路径')
    parser.add_argument('-i', '--image-dir', type=str, required=True,
                       help='图像目录路径')
    parser.add_argument('-img', '--single-image', type=str, default=None,
                       help='单张图像文件名（可选）')
    parser.add_argument('-o', '--output', type=str, default='visualization_results/gt',
                       help='输出目录')
    parser.add_argument('-n', '--max-images', type=int, default=None,
                       help='最多处理的图像数（可选）')
    parser.add_argument('--show-area', action='store_true',
                       help='是否显示标注框面积')
    parser.add_argument('--stats', action='store_true',
                       help='只打印统计信息，不进行可视化')
    
    args = parser.parse_args()
    
    # 打印统计信息
    if args.stats:
        print_statistics(args.annotation)
        return
    
    # 可视化
    if args.single_image:
        # 单张图像
        image_path = os.path.join(args.image_dir, args.single_image)
        visualize_single_image(image_path, args.annotation, args.output, 
                              args.show_area)
    else:
        # 整个数据集
        visualize_dataset(args.image_dir, args.annotation, args.output,
                         args.max_images, args.show_area)


if __name__ == '__main__':
    main()

