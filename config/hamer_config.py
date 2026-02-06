import torch
import os

class Config:
    root_dir = '/home/pt/fbs/model'
    ckpt_path = os.path.join(root_dir, 'hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt')
    model_cfg = os.path.join(root_dir, 'hamer/_DATA/hamer_ckpts/model_config.yaml')
    onnx_path = os.path.join(root_dir, 'hamer/_DATA/hamer_ckpts/onnx/hamer_inferpy.onnx')
    use_onnx = False

hamer_opt = Config()
