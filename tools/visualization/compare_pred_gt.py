"""
同时可视化预测框和真实标注框
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from engine.core import YAMLConfig

# ChestX_Det类别定义
CHEST_CATEGORIES = {
    1: 'Atelectasis', 2: 'Calcification', 3: 'Cardiomegaly',
    4: 'Consolidation', 5: 'DiffuseNodule', 6: 'Effusion',
    7: 'Emphysema', 8: 'Fibrosis', 9: 'Fracture',
    10: 'Mass', 11: 'Nodule', 12: 'Pleural Thickening',
    13: 'Pneumothorax'
}


def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()


def draw_comparison(image_path, gt_annotations, pred_results, categories_dict,
                   thrh=0.3, font_size=15):
    """
    绘制预测框和真实框的对比
    
    GT框使用绿色实线
    预测框使用红色虚线（通过多个矩形模拟）
    """
    im = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(im)
    font = get_font(font_size)
    
    # 1. 绘制Ground Truth (绿色实线)
    for ann in gt_annotations:
        x, y, width, height = ann['bbox']
        box = [x, y, x + width, y + height]
        category_id = ann['category_id']
        category_name = categories_dict.get(category_id, f"ID_{category_id}")
        
        # 绘制绿色框
        draw.rectangle(box, outline=(0, 255, 0), width=3)
        
        # 绘制标签
        text = f"GT: {category_name}"
        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 文本背景
        bg_box = [box[0], box[1] - text_height - 5, 
                  box[0] + text_width + 4, box[1]]
        draw.rectangle(bg_box, fill=(0, 255, 0))
        draw.text((box[0] + 2, box[1] - text_height - 5), text, 
                 fill=(0, 0, 0), font=font)
    
    # 2. 绘制预测结果 (红色实线)
    if pred_results:
        labels, boxes, scores = pred_results
        
        # 过滤低置信度
        mask = scores > thrh
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]
        
        for label, box, score in zip(labels, boxes, scores):
            category_id = label.item() + 1  # 转换为COCO ID
            category_name = categories_dict.get(category_id, f"ID_{category_id}")
            
            # 绘制红色框
            draw.rectangle(list(box), outline=(255, 0, 0), width=3)
            
            # 绘制标签
            text = f"Pred: {category_name} {score:.2f}"
            text_bbox = draw.textbbox((box[0], box[3]), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 文本背景
            bg_box = [box[0], box[3] + 2, 
                      box[0] + text_width + 4, box[3] + text_height + 7]
            draw.rectangle(bg_box, fill=(255, 0, 0))
            draw.text((box[0] + 2, box[3] + 4), text, 
                     fill=(255, 255, 255), font=font)
    
    return im


def load_model(config_path, checkpoint_path, device):
    """加载模型"""
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(device)
    model.eval()
    return model


def predict_image(model, image_path, device):
    """对单张图像进行预测"""
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(im_data, orig_size)
    
    labels = output[0]['labels'].cpu()
    boxes = output[0]['boxes'].cpu()
    scores = output[0]['scores'].cpu()
    
    return labels, boxes, scores


def main():
    parser = argparse.ArgumentParser(description='同时可视化预测框和真实标注框')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('-r', '--resume', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('-a', '--annotation', type=str, required=True,
                       help='COCO标注文件路径')
    parser.add_argument('-i', '--image-dir', type=str, required=True,
                       help='图像目录路径')
    parser.add_argument('-img', '--single-image', type=str, default=None,
                       help='单张图像文件名（可选）')
    parser.add_argument('-o', '--output', type=str, 
                       default='visualization_results/pred_vs_gt',
                       help='输出目录')
    parser.add_argument('-t', '--thrh', type=float, default=0.3,
                       help='预测置信度阈值')
    parser.add_argument('-n', '--max-images', type=int, default=10,
                       help='最多处理的图像数')
    
    args = parser.parse_args()
    
    # 加载标注
    print("加载标注文件...")
    with open(args.annotation, 'r') as f:
        coco_data = json.load(f)
    
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 加载模型
    print("加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.config, args.resume, device)
    
    os.makedirs(args.output, exist_ok=True)
    
    # 处理图像
    if args.single_image:
        images_to_process = [(None, {'file_name': args.single_image})]
        for img_id, info in img_id_to_info.items():
            if info['file_name'] == args.single_image:
                images_to_process = [(img_id, info)]
                break
    else:
        images_to_process = list(img_id_to_info.items())[:args.max_images]
    
    print(f"\n处理 {len(images_to_process)} 张图像...")
    
    for img_id, img_info in tqdm(images_to_process):
        image_path = os.path.join(args.image_dir, img_info['file_name'])
        
        if not os.path.exists(image_path):
            continue
        
        # 获取GT标注
        gt_annotations = img_id_to_anns.get(img_id, [])
        
        # 获取预测结果
        pred_results = predict_image(model, image_path, device)
        
        # 绘制对比
        result_image = draw_comparison(image_path, gt_annotations, pred_results,
                                       categories_dict, args.thrh)
        
        # 保存
        output_file = os.path.join(args.output, f"compare_{img_info['file_name']}")
        result_image.save(output_file)
    
    print(f"\n✅ 完成！结果保存在: {args.output}")
    print("\n图例:")
    print("  • 绿色框 = Ground Truth (真实标注)")
    print("  • 红色框 = Prediction (模型预测)")


if __name__ == '__main__':
    main()

