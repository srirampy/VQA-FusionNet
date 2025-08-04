import timm
import torch
from src.config import device

def get_vit_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    return model.to(device)
