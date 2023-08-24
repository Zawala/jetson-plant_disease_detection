from imports import *
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
        """
        Function loads the yolo5 model from PyTorch Hub.
        """
        #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        #model = torch.load('model_a.pth')
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
        image_np = np.array([self.process_image(Image.open(frame))])
        image = torch.FloatTensor(image_np)
        self.model.eval()
        output = self.model.forward(Variable(image))
        pobabilities = torch.exp(output).data.numpy()[0]
    

        top_idx = np.argsort(pobabilities)[-topk:][::-1] 
        top_class = [self.idx_to_class[x] for x in top_idx]
        top_probability = pobabilities[top_idx]

        return top_probability, top_class
    
    def draw_predictions(self,image_raw):
        
        # Load an image
        image = Image.open(image_raw)

        # Create a drawing object
        draw = ImageDraw.Draw(image)

        # Define the coordinates of the top-left and bottom-right corners of the rectangle
        top_left = (50, 50)
        bottom_right = (150, 150)

#       Define the color of the rectangle (in RGB format)
        color = (0, 255, 0)

#       Draw the rectangle on the image
        draw.rectangle([top_left, bottom_right], outline=color, width=2)

#       Define the text to be displayed and its position

        text = self.score_frame(image_raw)
        text_position = (50, 40)

#           Define the font and size of the text
        font = ImageFont.truetype('arial.ttf', size=12)

                # Draw the text on the image
        draw.text(text_position, text[1][0], fill=color, font=font)

#        Display the image with the rectangle and text
        image.show()
         

    def process_image(self, image):
        
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
    

    def imshow(self, image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
    
        ax.imshow(image)
    
        return ax
    
    
    def __call__(self):
       print(self.draw_predictions('image (2).JPG'))


#link = sys.argv[1]
output_file = 'Labeled_Video.avi'
a = ObjectDetection()
a()