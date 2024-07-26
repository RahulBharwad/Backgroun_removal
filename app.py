from flask import Flask, request, send_file, render_template_string
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import os 

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

model = deeplabv3_resnet101(pretrained=True)
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0), image

def postprocess_output(output, original_image):
    output_predictions = output['out'].argmax(1).squeeze(0).detach().cpu().numpy()
    mask = (output_predictions == 15).astype(np.uint8)  # Class 15 is 'person' in COCO dataset

    mask = Image.fromarray(mask).resize(original_image.size, Image.NEAREST)
    mask = np.array(mask)

    white_background = np.ones_like(np.array(original_image)) * 255
    result = np.array(original_image) * np.expand_dims(mask, axis=2) + white_background * (1 - np.expand_dims(mask, axis=2))
    return Image.fromarray(result.astype(np.uint8))

def remove_background(image_path, output_path):
    input_tensor, original_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
    result_image = postprocess_output(output, original_image)
    result_image.save(output_path)

@app.route('/')
def index():
    return render_template_string('''
    <!doctype html>
    <title>Background Remover</title>
    <h1>Upload an image to remove background</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    ''')

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        input_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + file.filename)
        file.save(input_image_path)
        remove_background(input_image_path, output_image_path)
        return send_file(output_image_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=False)