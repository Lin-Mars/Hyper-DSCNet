# Hyper-DSCNet: Integrating Radiographic Details and Syndromic Context via Hypergraph-Enhanced Dual-Stream Learning for Chest X-Ray Lesion Detection

## Weights

[Hugging Face](https://huggingface.co/Lin-Mars/Hyper-DSCNet)

## Training

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/yaml/deim_dfine_hgnetv2_s.yml --use-amp --seed=0
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7778 --nproc_per_node=1 train.py -c configs/yaml/deim_dfine_hgnetv2_s.yml --test-only -r dfine-s-hyperACE+p2.pth
```
