# MAF-YOLO

## 训练你的数据集

### 环境配置
```
conda create -n MAF-YOLO python==3.8
conda activate MAF-YOLO
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

```

### 数据集文件结构
```
├── data
│   ├── images
│   │   ├── train
│   │   └── val
│   ├── labels
│   │   ├── train
│   │   ├── val
```
### data.yaml 配置
```shell
train: data/images/train # 训练集路径
val: data/images/val # 验证集路径
is_coco: False
nc: 3  # 设置为你的类别数量
names: ["car","person","bike"] #类别名称
```

### Acknowledgements
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
* [https://github.com/yang-0201/YOLOv6_pro](https://github.com/yang-0201/YOLOv6_pro)
