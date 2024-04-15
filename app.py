# import numpy as np
from numpy import asarray
from flask import Flask, request, jsonify, render_template, make_response, url_for
# import pickle
from PIL import Image 
# import torchvision
# from torchvision import datasets, transforms
from werkzeug.utils import secure_filename
import io
import torch
#import pdfkit
import base64
from io import BytesIO
import torchvision.transforms.functional as TF
from flask import render_template_string
import os
from pathlib import Path
import tempfile
import ast

import torch
from torchvision import transforms
from PIL import Image
import io
import os
from flask_cors import CORS

# 设置Flask应用和上传目录
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
food_descriptions = 'food_descriptions.txt'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 您之前的模型加载代码
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('model/final_modelF11.pth', map_location=device)
model = model.to(device)
model.eval()

classFile = 'class_to_idx.txt'
idx_to_class = {}

# 打开文件并读取内容
# with open(classFile, 'r') as file:
with open(classFile, 'r') as file:
    idx_to_class = {}  # 确保有一个字典来存储索引到类名的映射
    for line in file:
        # 移除行末的换行符
        line = line.strip()
        # 分割索引和类名
        parts = line.split(':')  # 使用 split 方法一次性分割字符串
        if len(parts) == 2:  # 确保行的格式正确
            idx, class_name = parts
            # 去除类名中的下划线
            class_name_no_underscore = class_name.replace('_', ' ')
            idx_to_class[int(idx)] = class_name_no_underscore
        # else:
        #     idx = line.split(':')[0]  # 获取索引
        #     class_name = line.split(':')[1]  # 获取类名
        #     idx_to_class[int(idx)] = class_name

def load_food_descriptions():
    descriptions = {}
    with open(food_descriptions, 'r') as file:
        for line in file:
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                idx, description = parts
                descriptions[int(idx)] = description
    return descriptions

food_descriptions = load_food_descriptions()

# 预测函数
def predict_image(image_path):
    pId = 0

    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


    # 加载图像
    image = Image.open(image_path)
    transformed_image = transform(image)
    batch_t = torch.unsqueeze(transformed_image, 0)
    batch_t = batch_t.to(device)

    # 预测
    with torch.no_grad():
        output, global_class_scores, regional_class_scores = model(batch_t)
        _, predicted = torch.max(output, 1)

    # 将类别索引转换为实际类名
    # idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    predicted_class = idx_to_class[predicted.item()]
    pId = predicted.item()
    return predicted_class, pId

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        
        # Assume prediction function is correct and returns a string
        predicted_class, pId = predict_image(image_path)
        
        os.remove(image_path)
        description = food_descriptions.get(int(pId), "Description not found.")

        return jsonify({'result': predicted_class, 'description': description})

    return jsonify({'error': 'Unknown error occurred'}), 500

@app.route('/test')
def test_route():
    return 'Flask is working!'

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/qa')
def qa():
    return render_template('qa.html')

if __name__ == '__main__':
    app.run(debug=True)
