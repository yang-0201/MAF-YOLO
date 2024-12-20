# MAF-YOLO
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2407.04381-B31B1B.svg)](https://arxiv.org/abs/2407.04381)
## UPDATES 🔥
- 2024/6/25: Add MAF-YOLO
- 2024/11/11: Add [MAF-YOLOv2](https://github.com/yang-0201/MAF-YOLOv2). This model achieves exceptionally high parameter efficiency and has reached SOTA performance among all YOLO variants. 

This is the official MegEngine implementation of MAF-YOLO, from the following PRCV2024 (Oral) paper:

[Multi-Branch Auxiliary Fusion YOLO with Re-parameterization Heterogeneous Convolutional for accurate object detection.](https://arxiv.org/abs/2407.04381)

Article Interpretation: [集智书童](https://mp.weixin.qq.com/s/2Kdp0VWP-AKK70yq7TExFg)

<div align="center">
    <a href="./">
        <img src="./picture/result.png" width="79%"/>
    </a>
</div>

## Performance
MS COCO

| Model                                                  | Test Size | #Params | FLOPs | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sup>test2017</sup> | AP<sub>50</sub><sup>test2017</sup> |epoch |
|:-------------------------------------------------------|:----:|:-------:|:-----:|:----------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
| [MAF-YOLO-N](https://github.com/yang-0201/MAF-YOLO/releases/download/v1.0.0/MAFYOLOn.pt) |   640  |  3.8M   | 10.5G |      42.4%       |             58.9%          | 42.1% | 58.6% | 300 | 
| [MAF-YOLO-S](https://github.com/yang-0201/MAF-YOLO/releases/download/v1.0.0/MAFYOLOs.pt) |   640  |  8.6M   | 25.5G |      47.4%       |             64.3%         | 47.2% | 64.0%    |300 |
| [MAF-YOLO-M](https://github.com/yang-0201/MAF-YOLO/releases/download/v1.0.0/MAFYOLOm.pt) |   640  |  23.7M  | 76.7G |      51.2%       |             68.5%           | 50.9% | 68.1%   |300 |

Our second version of the work to improve on YOLOv10 has yielded preliminary results that compare favorably with both YOLOv10 and YOLO11.

You can see it at this link: [MAF-YOLOv2](https://github.com/yang-0201/MAF-YOLOv2)!
    
| Model                                                  | Test Size | #Params | FLOPs | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sup>test2017</sup> | AP<sub>50</sub><sup>test2017</sup> |epoch |
|:-------------------------------------------------------|:----:|:-------:|:-----:|:----------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
| YOLOv10n |   640  |  2.3M   | 6.7G |      38.5%       |             53.8%          | 38.7% | 54.4% | 500 | 
| MAF-YOLOv10n |   640  |  2.2M   | 7.2G |      42.3%       |             58.5%          | 42.3% | 58.4% | 500 | 
| YOLOv10s |   640  |  7.2M   | 21.6G |      46.3%       |             63.0%         | 45.9% | 62.4%    |500 |
| MAF-YOLOv10s |   640  |  7.1M   | 25.3G |      48.9%       |             65.9%         | 48.8% | 65.5%    |500 |
| YOLOv10m |   640  |  15.4M   | 59.1G |      51.1%       |             68.1%         | 51.2% | 68.0%    |500 |
| MAF-YOLOv10m |   640  |  15.3M   | 65.2G |      52.7%       |             69.5%         | - | -    |500 |
## Installation

```
conda create -n mafyolo python==3.8
conda activate mafyolo
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
## Evaluation
```python
# evaluate MAF-YOLOn
python tools/eval.py --weights MAFYOLOn.pt --data data/coco.yaml

# evaluate MAF-YOLOs
python tools/eval.py --weights MAFYOLOs.pt --data data/coco.yaml --reproduce_640_eval

# evaluate MAF-YOLOm
python tools/eval.py --weights MAFYOLOm.pt --data data/coco.yaml --reproduce_640_eval
```

## Train
Single GPU training
```python
# Loading pre-trained weight to train MAFYOLOn
python tools/train.py --conf configs/pretrain/MAF-YOLO-n-pretrain.py --data data/coco.yaml --device 0

# Training MAFYOLOn from scratch
python tools/train.py --conf configs/MAF-YOLO-n.py --data data/coco.yaml --device 0
```
Multiple GPU training
```python
# Training MAFYOLOn from scratch with multiple GPU
python -m torch.distributed.run --nproc_per_node 4 --master_port 9527 python tools/train.py --conf configs/MAF-YOLO-n.py --data data/coco.yaml --device 0,1,2,3
```
## Dataset Configuration
<details><summary> <b>Dataset file structure</b> </summary>

```
├── data
│   ├── images
│   │   ├── train
│   │   └── val
│   ├── labels
│   │   ├── train
│   │   ├── val
```
</details>

<details><summary> <b>data.yaml</b> </summary>

```shell
train: data/images/train 
val: data/images/val 
is_coco: False
nc: 3  
names: ["car","person","bike"] 
```
</details>

## Citation

If our code or model is helpful to your work, please cite our paper. We would be very grateful!
```BibTeX
@InProceedings{10.1007/978-981-97-8858-3_34,
author="Yang, Zhiqiang
and Guan, Qiu
and Zhao, Keer
and Yang, Jianmin
and Xu, Xinli
and Long, Haixia
and Tang, Ying",
editor="Lin, Zhouchen
and Cheng, Ming-Ming
and He, Ran
and Ubul, Kurban
and Silamu, Wushouer
and Zha, Hongbin
and Zhou, Jie
and Liu, Cheng-Lin",
title="Multi-branch Auxiliary Fusion YOLO with Re-parameterization Heterogeneous Convolutional for Accurate Object Detection",
booktitle="Pattern Recognition and Computer Vision",
year="2025",
publisher="Springer Nature Singapore",
address="Singapore",
pages="492--505",
abstract="Due to the effective performance of multi-scale feature fusion, Path Aggregation FPN (PAFPN) is widely employed in YOLO detectors. However, it cannot efficiently and adaptively integrate high-level semantic information with low-level spatial information simultaneously. We propose a new model named MAF-YOLO in this paper, which is a novel object detection framework with a versatile neck named Multi-Branch Auxiliary FPN (MAFPN). Within MAFPN, the Superficial Assisted Fusion (SAF) module is designed to combine the output of the backbone with the neck, preserving an optimal level of shallow information to facilitate subsequent learning. Meanwhile, the Advanced Assisted Fusion (AAF) module deeply embedded within the neck conveys a more diverse range of gradient information to the output layer. Furthermore, our proposed Re-parameterized Heterogeneous Efficient Layer Aggregation Network (RepHELAN) module ensures that both the overall model architecture and convolutional design embrace the utilization of heterogeneous large convolution kernels. Therefore, this guarantees the preservation of information related to small targets while simultaneously achieving the multi-scale receptive field. Finally, taking the nano version of MAF-YOLO for example, it can achieve 42.4{\%} AP on COCO with only 3.76M learnable parameters and 10.51G FLOPs, and approximately outperforms YOLOv8n by about 5.1{\%}. The source code of this work is available at: https://github.com/yang-0201/MAF-YOLO.",
isbn="978-981-97-8858-3"
}
```
or

```BibTeX
@article{yang2024multi,
  title={Multi-Branch Auxiliary Fusion YOLO with Re-parameterization Heterogeneous Convolutional for accurate object detection},
  author={Yang, Zhiqiang and Guan, Qiu and Zhao, Keer and Yang, Jianmin and Xu, Xinli and Long, Haixia and Tang, Ying},
  journal={arXiv preprint arXiv:2407.04381},
  year={2024}
}
```

## Acknowledgements

* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
* [https://github.com/yang-0201/YOLOv6_pro](https://github.com/yang-0201/YOLOv6_pro)
