# Learning for Long-Tailed Visual Recognition of 3D Point Clouds
VLR Project Work

## Usage
```bash
# To train on ModelNet10 dataset:
python main/train.py  --cfg configs/modelNet10.yaml     

# To validate with the best model:
python main/valid.py  --cfg configs/modelNet10.yaml

# To debug with CPU mode:
python main/train.py  --cfg configs/modelNet10.yaml   CPU_MODE True
```

We have used the official repo of BBN [https://github.com/megvii-research/BBN] as a reference for our implementation
