import cv2
#import tk_stopwatch as tks
import time
import sys
import tty
import termios
import multiprocessing
import easyocr
from PIL import Image
vidCap = cv2.VideoCapture(0)
vidCap.set(3,640) #set width
vidCap.set(4,480) #set height
vidCap.set(10,100) #set
minArea = 500


plantsCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml") #model for licence plates
timeStamp = [0,0,False]

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def plantView(ROI,image):
    cv2.imshow("vid",ROI)
    detection = easyocr.Reader(['en'])
    ocrResult = detection.readtext(ROI)
    text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
    cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # display the license plate and the output image
    cv2.imshow('Image', image)
    cv2.waitKey(0)

#x = multiprocessing.Process(target=tks.app.mainloop(), args=(1,))
while True:
    success, img = vidCap.read() #capture image by image

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale

    plates = plantsCascade.detectMultiScale(imgGray, 1.1, 4)  # detect all the plates


        #x.start()
    for x, y, w, h in plates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 250, 0), 2)  # draw a rectangle around the plates
            imgROI = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgROI)
            if timeStamp[2] == False:
                timeStamp[0] = time.time()
                timeStamp[2] = True
            plantView(imgROI,img)



    img = Image.fromarray(imgGray)
    img.show()

    
    key = getch()
    if key == 'q':

        if timeStamp[2]:
            timeStamp[1] = time.time()
            timeStamp[2] = False
        print(f'end {round(timeStamp[1])} minutes(m)')
        print(f' start{round(timeStamp[0])} minutes(m)')
        timeAccumulated = timeStamp[1]-timeStamp[0]
        #stopwatchTime = tks.app.seconds
        print(f'Finished in {round(timeAccumulated, 2)} minutes(m)')
        #print(f'Finished in {round(stopwatchTime, 2)} minutes(m)')
        print(f'total cost {round(timeAccumulated*(1/60),2)} dollars')

        break



