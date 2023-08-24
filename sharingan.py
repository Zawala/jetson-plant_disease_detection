#import pyzed.sl as sl
from collections import OrderedDict
#from datetime import datetime
from torch.autograd import Variable
import time
from PIL import Image
#import matplotlib.pyplot as plt
import copy
import cv2
import pyzed.sl as sl
import numpy as np
import os

#=============================#
#pytorch imports              #
#=============================#
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, datasets, transforms
from torchvision.models import ResNet18_Weights

#from train import Model as plantmodel

class ObjectDetection:
    """
    The class performs generic object detection on a video file.
    It uses yolo5 pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot boxes on objects along with labels.
    Upcoming Features:
    """
    def __init__(self):
        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """
        self.model, self.class_to_idx = self.load_model('res18tomato_checkpoint.pth')

        self.idx_to_class = { v : k for k,v in self.class_to_idx.items()}
        #self.model.conf = 0.4 # set inference threshold at 0.3
        #self.model.iou = 0.3 # set inference IOU threshold at 0.3
        #self.model.classes = [0] # set model to only detect "Person" class
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, filepath):
        
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
    
    def score_frame(self, frame, topk=1):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.

        """
        pil_image = Image.fromarray(frame).convert('RGB')
        image_np = np.array([self.process_image(pil_image)])

        image = torch.FloatTensor(image_np)
        self.model.eval()
        output = self.model.forward(Variable(image))
        pobabilities = torch.exp(output).data.numpy()[0]

        top_idx = np.argsort(pobabilities)[-topk:][::-1] 
        top_class = [self.idx_to_class[x] for x in top_idx]
        top_probability = pobabilities[top_idx]

        return top_probability, top_class
    
    def draw_predictions(self):
        # Create a Camera object
        zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD2K  # Use HD2K video mode for higher quality images
        init_params.camera_fps = 60  # Set fps at 60

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Create Mat objects for left and right images
        image_left = sl.Mat()
        image_right = sl.Mat()

    # Get screen size
        cv2.namedWindow("ZED", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("ZED", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        screen_width = cv2.getWindowImageRect("ZED")[2]



    # Capture new images until 'q' is pressed
        key = ''
        while key != ord('q'):
                # Grab an image
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Retrieve left and right images
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                # Combine left and right images horizontally
                combined_image = np.hstack((self.label_frame(image_left.get_data()), self.label_frame(image_right.get_data())))
                # Resize combined image to fit screen
                #combined_image = cv2.resize(combined_image, (screen_width, int(screen_width * combined_image.shape[0] / combined_image.shape[1])))
                # Display combined image with OpenCV
                cv2.imshow("ZED", combined_image)
                key = cv2.waitKey(1)


        # Close the camera and destroy window
        zed.close()
        cv2.destroyAllWindows()

    def label_frame(self,frame):
        
        # Preprocess the frame for AI inference

         # Perform AI inference on the frame
            prediction = self.score_frame(frame)

        # Draw boxes on regions of interest with some text
            print(prediction[0][0])
            frame_height, frame_width = frame.shape[:2]
            rect_width, rect_height = 600, 600

            x1 = (frame_width - rect_width) // 2
            y1 = (frame_height - rect_height) // 2
            x2 = x1 + rect_width
            y2 = y1 + rect_height

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print(prediction)
            try:
                if prediction[0][0]>0.6:
                    label=prediction[1][0]
                    
                else:
                    label='unsure'
            except Exception as e:
                pass
            accuracy=int(prediction[0][0]*100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {accuracy}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 8)

           # Display the resulting frame
            return frame

    def process_image(self, image):
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2
        image = image.crop((left, top, right, bottom))
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
    

    #def imshow(self, image, ax=None, title=None):
        #if ax is None:
       #     fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
      #  image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
     #   mean = np.array([0.485, 0.456, 0.406])
    #    std = np.array([0.229, 0.224, 0.225])
   #     image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
  #      image = np.clip(image, 0, 1)
    
 #       ax.imshow(image)
    
 #       return ax
    
    
    def __call__(self):
       self.draw_predictions()


a = ObjectDetection()
a()