# Hyper-DSCNet: Integrating Radiographic Details and Syndromic Context via Hypergraph-Enhanced Dual-Stream Learning for Chest X-Ray Lesion Detection

## Weights

Model weights and pretrained weights are available on [Hugging Face](https://huggingface.co/Lin-Mars/Hyper-DSCNet).

## Datasets

- [ChestX_Det](https://huggingface.co/datasets/Lin-Mars/ChestX_Det)
- [amia-public-challenge-2024](https://www.kaggle.com/competitions/amia-public-challenge-2024/data)

## Training

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/yaml/deim_dfine_hgnetv2_s.yml --use-amp --seed=0
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7778 --nproc_per_node=1 train.py -c configs/yaml/deim_dfine_hgnetv2_s.yml --test-only -r Hyper-DSCNet.pth
```

## Project Structure

```
├── train.py
├── requirements.txt
├── amia-public-challenge-2024/
│   ├── annotations/
│   ├── test/
│   └── train/
├── ChestX_Det/
│   ├── annotations/
│   └── images/
├── configs/
│   ├── base/
│   ├── cfg/
│   ├── cfg-improve/
│   │   └── Hyper-DSCNet.yaml
│   ├── dataset/
│   ├── dfine/
│   └── yaml/
│       └── deim_dfine_hgnetv2_s.yml
├── engine/
│   ├── backbone/
│   ├── core/
│   ├── data/
│   ├── deim/
│   ├── misc/
│   ├── modules/
│   │   ├── tasks.py
│   │   ├── custom_nn/
│   │   └── ultralytics_nn/
│   ├── optim/
│   └── solver/
├── compile_module/
├── tools/
│   ├── inference/
│   └── visualization/
└── weight/
    └── hgnetv2/
        └── PPHGNetV2_B0_stage1.pth
```
