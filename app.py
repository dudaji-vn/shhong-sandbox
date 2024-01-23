# app.py

import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, send_file

import threading
import time
import logging
import random
import queue
from multiprocessing import Pool

import pytorch_client as pc
model = pc.get_model()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def _get_prediction(tensor):
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes).to("cuda")
    return _get_prediction(tensor)


app = Flask(__name__)


@app.route('/preprocess', methods=['POST'])
def preprocess():
    if request.method == 'POST':
        file = request.files['file']
        image_bytes = file.read()
        tensor = transform_image(image_bytes)
        buff = io.BytesIO()
        torch.save(tensor, buff)
        buff.seek(0)
        return send_file(buff, mimetype='application/octet-stream')


@app.route('/predict', methods=['POST'])
def predict2():
    buff = request.data
    tensor = torch.load(io.BytesIO(buff)).to('cuda')
    _get_prediction(tensor)
    return jsonify({})


@app.route('/ignoreme', methods=['POST'])
def predict():
    # ignore me
    if request.method == 'POST':
        file = request.files['file']
        image_bytes = file.read()
        get_prediction(image_bytes=image_bytes)
        return jsonify({})


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)

