import os
import torch
from torchvision import transforms
from PIL import Image
import time


def get_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval().to("cuda")
    return model


def create_batch(filename):
    img = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(255), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), transforms.Normalize( 
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = preprocess(img)
    batch = tensor.unsqueeze(0)
    return batch


def run(model, filename="img1.jpg"):
    batch = create_batch(filename).to("cuda")
    outputs = model.forward(batch)
    _, y_hat = outputs.max(1)
    return y_hat.item()

