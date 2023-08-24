from __future__ import division, print_function
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.models import ResNet18_Weights

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
def load_model(filepath):
        
        checkpoint = torch.load(filepath)
        model = models.resnet18()
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512, 512)),
                          ('relu', nn.ReLU()),
                          #('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(512, 39)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

        # Replacing the pretrained model classifier with our classifier
        model.fc = classifier
        model.load_state_dict(checkpoint['state_dict'])
        return model, checkpoint['class_to_idx']
model, class_to_idx = load_model('res18tomato_checkpoint.pth')

idx_to_class = { v : k for k,v in class_to_idx.items()}
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def process_image(image):
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2
        image = image.crop((left, top, right, bottom))
        print(image.size)
        size = 256, 256
        image.thumbnail(size, Image.Resampling.LANCZOS)
        image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
        npImage = np.array(image)
        npImage = npImage/255.
        
        imgA = npImage[:,:,0]
        imgB = npImage[:,:,1]
        imgC = npImage[:,:,2]
    
        imgA = (imgA - 0.485)/(0.229) 
        imgB = (imgB - 0.456)/(0.224)
        imgC = (imgC - 0.406)/(0.225)
        
        npImage[:,:,0] = imgA
        npImage[:,:,1] = imgB
        npImage[:,:,2] = imgC
    
        npImage = np.transpose(npImage, (2,0,1))
    
        return npImage

def score_frame(pil_image):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.

        """
        topk=1
        image_np = np.array([process_image(Image.open(pil_image))])

        image = torch.FloatTensor(image_np)
        model.eval()
        output = model.forward(Variable(image))
        pobabilities = torch.exp(output).data.numpy()[0]

        top_idx = np.argsort(pobabilities)[-topk:][::-1] 
        top_class = [idx_to_class[x] for x in top_idx]
        top_probability = pobabilities[top_idx]

        return top_probability, top_class
    

def model_predict(img_path, model):
        
        # Preprocess the frame for AI inference

         # Perform AI inference on the frame
            prediction = score_frame(img_path)

        # Draw boxes on regions of interest with some text
            print(prediction[0][0])
            print(prediction)
            #try:
            #    if prediction[0][0]>0.7:
            label=prediction[1][0]
                    
            #    else:
            #        label='background'
            #except Exception as e:
            #    pass
           
            return label
        
    
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
