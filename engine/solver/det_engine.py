"""   
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.     
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)   
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""  
  

import os, sys   
import math  
import json 
import numpy as np 
from typing import Iterable
from tqdm import tqdm
from prettytable import PrettyTable 
# from pycocotools.coco import COCO  
# from pycocotools.cocoeval import COCOeval    
from tidecv import TIDE, datasets
     
import torch   
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler    

from ..optim import ModelEMA, Warmup  
from ..data import CocoEvaluator 
from ..misc import MetricLogger, MetricLogger_progress, SmoothedValue, dist_utils, plot_sample     
from ..modules.ops import Profile  
from ..modules.utils import TQDM, RANK

TIME_DEBUG = False
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 

# 训练单个 epoch
# self_lr_scheduler: 是否使用自定义学习率调度器
# lr_scheduler: 学习率调度器实例 
# model: 训练的 PyTorch 模型
# criterion: 损失计算函数
# data_loader: 训练数据加载器 
# optimizer: 优化器
# device: 训练设备（CPU 或 GPU）
# epoch: 当前 epoch 计数    
# max_norm: 梯度裁剪的最大范数     
# **kwargs: 其他参数，例如日志记录等
def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,     
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()  # 设置模型为训练模式   
    criterion.train()  # 设置损失函数为训练模式     

    print_freq = kwargs.get('print_freq', 10)  # 日志打印频率   
    writer: SummaryWriter = kwargs.get('writer', None)  # TensorBoard 记录器
    ema: ModelEMA = kwargs.get('ema', None)  # 指数移动平均模型  
    scaler: GradScaler = kwargs.get('scaler', None)  # 混合精度训练的梯度缩放器 
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)  # 预热学习率调度器
    plot_train_batch_freq = kwargs.get('plot_train_batch_freq', 12)  
    output_dir = kwargs.get('output_dir', None)    
    epoches = kwargs.get('epoches', -1) # 总的训练次数
    verbose_type = kwargs.get('verbose_type', 'origin') # 显示方式
    header = 'Epoch: {}/{}'.format(epoch, epoches)  # 训练过程的日志标题
  
    cur_iters = epoch * len(data_loader)  # 计算当前 epoch 的起始迭代数   
    
    if verbose_type == 'origin':
        metric_logger = MetricLogger(delimiter="  ")  # 记录训练过程中的度量信息   
    else:  
        metric_logger = MetricLogger_progress(delimiter="  ")  # 记录训练过程中的度量信息     
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录学习率变化
    pbar = enumerate(metric_logger.log_every(data_loader, print_freq if verbose_type == 'origin' else 1, header))  
    
    dt = [  
        Profile(device=device), 
        Profile(device=device),     
        Profile(device=device),
        Profile(device=device),  
        Profile(device=device)  
    ] 
     
    for i, (samples, targets) in pbar:     
        if epoch % plot_train_batch_freq == 0 and i == 0:   
            if data_loader.dataset.remap_mscoco_category:
                plot_sample((samples, targets), data_loader.dataset.category2name, output_dir / f"train_batch_{epoch}.png", data_loader.dataset.label2category) 
            else:
                plot_sample((samples, targets), data_loader.dataset.category2name, output_dir / f"train_batch_{epoch}.png")
        with dt[0]:     
            samples = samples.to(device, non_blocking=True)  # 将输入数据移动到指定设备
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]  # 目标数据也移动到设备
   
        global_step = epoch * len(data_loader) + i  # 计算全局训练步数
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))  # 训练元数据   
    
        # 使用混合精度训练 
        if scaler is not None:
            with dt[1]:
                with torch.autocast(device_type=str(device), cache_enabled=True):     
                    outputs = model(samples, targets=targets) 
  
            # 处理异常情况，避免 NaN 或 Inf 影响训练
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():    
                print(f"[WARNING] NaN/Inf in pred_boxes at epoch {epoch} step {i}, skipping batch")
                # 修复被 NaN 前向传播污染的 BatchNorm running stats
                for module in model.modules():
                    if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
                        if torch.isnan(module.running_mean).any() or torch.isnan(module.running_var).any():
                            module.running_mean.zero_()
                            module.running_var.fill_(1.0)
                            module.num_batches_tracked.zero_()
                continue
            
            with dt[2]:
            # 计算损失   
                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)
                loss = sum(loss_dict.values())  # 总损失    
   
            with dt[3]:  
                scaler.scale(loss).backward()  # 反向传播  
     
                # 进行梯度裁剪（如果 max_norm > 0）   
                if max_norm > 0:    
                    scaler.unscale_(optimizer) 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                scaler.step(optimizer)  # 更新参数
                scaler.update()  # 更新梯度缩放因子   
                optimizer.zero_grad()  # 清空梯度
        
        else:     
            with dt[1]:
                outputs = model(samples, targets=targets)  # 前向传播
            with dt[2]:
                loss_dict = criterion(outputs, targets, **metas)  # 计算损失
                loss: torch.Tensor = sum(loss_dict.values())  # 总损失
            with dt[3]:
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播   
    
                # 进行梯度裁剪
                if max_norm > 0: 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)   
                optimizer.step()  # 更新参数    
 
        with dt[4]:
            # 更新 EMA（指数移动平均）   
            if ema is not None:     
                ema.update(model) 
     
            # 更新学习率     
            if self_lr_scheduler:
                optimizer = lr_scheduler.step(cur_iters + i, optimizer) 
            else:   
                if lr_warmup_scheduler is not None:     
                    lr_warmup_scheduler.step()

            # 计算损失并检查是否异常  
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)     
            loss_value = sum(loss_dict_reduced.values())   
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))    
                print(loss_dict_reduced)     
                sys.exit(1)     

            # 记录日志
            metric_logger.update(loss=loss_value, **loss_dict_reduced)    
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])     
   
            # 记录到 TensorBoard    
            if writer and dist_utils.is_main_process() and global_step % 10 == 0:
                writer.add_scalar('Loss/total', loss_value.item(), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
                for k, v in loss_dict_reduced.items():   
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)    

    # 统计并打印训练结果 
    metric_logger.synchronize_between_processes()    
    print(GREEN, "Averaged stats:", metric_logger, RESET)  
    if TIME_DEBUG:
        time_data = [x.t / len(data_loader) for x in dt]
        print(RED + f"Data_to_Device:{time_data[0]:.6f}s Inference:{time_data[1]:.6f}s Loss:{time_data[2]:.6f}s Weight_Update:{time_data[3]:.6f}s" + RESET)    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, test_only=False, output_dir=None):
    # 评估函数，禁用梯度计算以减少内存占用并提高推理速度   
    model.eval()
    criterion.eval() 
    coco_evaluator.cleanup()
     
    metric_logger = MetricLogger_progress(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))    
    header = 'Test:'
  
    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys()) 
    # 获取 IoU 计算类型（如 'bbox' 或 'segm'） 
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    
    # 初始化时间记录器
    dt = [
        Profile(device=device),
        Profile(device=device)    
    ]    
  
    # 遍历数据集进行评估  
    coco_pred_json = []
    for samples, targets in metric_logger.log_every(data_loader, 1, header):     
        samples = samples.to(device, non_blocking=True)  # 将样本数据移动到指定设备（如 GPU）
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]  # 目标数据也移动到设备   

        with dt[0]: 
            outputs = model(samples)  # 前向传播，获取模型输出  
     
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)  # 获取原始目标尺寸
     
        with dt[1]:  
            results = postprocessor(outputs, orig_target_sizes)  # 通过后处理器处理模型输出
 
        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)} # 将评估结果与图像 ID 关联
        if coco_evaluator is not None:     
            coco_evaluator.update(res) # 更新 COCO 评估器  
            coco_pred_json.extend(list(coco_evaluator.coco_eval['bbox'].cocoDt.anns.values()))

    # gather the stats from all processes 在多进程环境下同步评估数据    
    metric_logger.synchronize_between_processes()
    print(GREEN + "Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # 统计耗时 
    if test_only:
        speed = dict(zip(['inference', 'postprocess'], (x.t / len(data_loader.dataset) * 1e3 for x in dt)))
        print('-'*20, f"Speed: {speed['inference']:.4f}ms inference, {speed['postprocess']:.4f}ms postprocess per image", '-'*20)
        print('-'*20, f"FPS(inference+postprocess): {1000 / (speed['inference'] + speed['postprocess']):.2f}", '-'*20)
   
    # accumulate predictions from all images 累积并计算最终评估结果     
    if coco_evaluator is not None:
        coco_evaluator.accumulate()    
        coco_evaluator.summarize()
        if test_only:
            print(ORANGE + f"Saving coco pred[{output_dir / 'pred.json'}] json...")
            with open(output_dir / 'pred.json', 'w') as f:     
                json.dump(coco_pred_json, f)  
            print("save success.")
    
            precisions = coco_evaluator.coco_eval['bbox'].eval['precision']
            cat_ids = coco_evaluator.coco_eval['bbox'].params.cat_ids   
            results_per_category = []    
            for idx, cat_id in enumerate(cat_ids):     
                t = []     
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = coco_evaluator.coco_gt.cats[cat_id]    
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:     
                    ap = np.mean(precision)
                else:
                    ap = float('nan') 
                t.append(f'{nm["name"]}')
                t.append(f'{round(ap, 3)}')    
    
                # indexes of IoU  @50 and @75  
                for iou in [0, 5]:
                    precision = precisions[iou, :, idx, 0, -1] 
                    precision = precision[precision > -1] 
                    if precision.size:
                        ap = np.mean(precision)     
                    else:
                        ap = float('nan')  
                    t.append(f'{round(ap, 3)}')
    
                # indexes of area of small, median and large  
                for area in [1, 2, 3]:
                    precision = precisions[:, :, idx, area, -1]
                    precision = precision[precision > -1]     
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')    
                    t.append(f'{round(ap, 3)}')
                results_per_category.append(list(t))

            # class_metrics = coco_evaluator.coco_eval['bbox'].extended_metrics['class_map']     
            model_metrice_table = PrettyTable() 
            model_metrice_table.title = "Model Metrice"   
            model_metrice_table.field_names = ['category', 'AP', 'AP_50', 'AP_75', 'AP_s', 'AP_m', 'AP_l']   
            for data in results_per_category:
                model_metrice_table.add_row(list(data))
    
            # 提取数值列（跳过第一列的类别名称） 
            numeric_data = [list(data)[1:] for data in results_per_category]   
            
            # 计算每列的平均值    
            avg_values = []   
            for col_idx in range(len(numeric_data[0])): 
                col_values = [float(row[col_idx]) for row in numeric_data if float(row[col_idx]) is not None]
                if col_values:    
                    avg_value = sum(col_values) / len(col_values)
                    avg_values.append(round(avg_value, 3))  # 保留3位小数
                else:
                    avg_values.append(0.0) 
            
            # 添加平均值行     
            all_row = ['all'] + avg_values  
            model_metrice_table.add_row(all_row)  
            
            print(model_metrice_table)

            # anno = COCO(str(data_loader.dataset.ann_file))  # init annotations api  
            # pred = anno.loadRes(str(output_dir / 'pred.json'))  # init predictions api
            # eval = COCOeval(anno, pred, 'bbox')
            # eval.evaluate()
            # eval.accumulate()
            # eval.summarize()

            try:    
                tide = TIDE()
                tide.evaluate_range(datasets.COCO(data_loader.dataset.ann_file), datasets.COCOResult(output_dir / 'pred.json'))
                tide.summarize()   
                tide.plot(out_dir=output_dir / 'tide_result') 
            except Exception as e:
                print(RED, 'TIDE failure... skip message:', e)   

    print(RESET)     
 
    stats = {}   
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}     
    if coco_evaluator is not None:   
        if 'bbox' in iou_types:     
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types: 
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()    
     
    return stats, coco_evaluator
