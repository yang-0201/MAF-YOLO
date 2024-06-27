import torch
import os.path as osp


ckpt_path = osp.join("MAF-YOLO-m.pt")
save_ckpt_dir = osp.join("MAF-YOLO-m-small.pt")
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
if ckpt.get('ema'):
    ckpt['model'] = ckpt['ema']  # replace model with ema
for k in ['optimizer', 'ema', 'updates']:  # keys
    ckpt[k] = None
ckpt['epoch'] = 300
ckpt['model'].half()  # to FP16
for p in ckpt['model'].parameters():
    p.requires_grad = False
torch.save(ckpt, save_ckpt_dir)