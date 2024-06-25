# YOLOv6n model
model = dict(
    type='YOLOv6n',
    pretrained="MAF-YOLO-n.pt",
    build_type='yaml',
    yaml_file='configs/yaml/MAF-YOLO-n.yaml',
    depth_multiple=0.33,
    width_multiple=0.25,
    head=dict(
        type='EffiDeHead',
        num_layers=3,
        anchors=1,
        strides=[8, 16, 32],
        iou_type='giou',
        use_dfl=True,
        reg_max=16 #if use_dfl is False, please set reg_max to 0
    )
)
solver = dict(
    optim='SGD',
    lr_scheduler='linear',
    lr0=0.01, #0.01 # 0.02
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    dy_label=5,
    dy_mixup=0.2,
    mask_refine = True,
    copy_paste = 0.05,
)
