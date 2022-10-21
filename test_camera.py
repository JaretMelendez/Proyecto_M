"""
Jiménez Cervantes José de Jesús 215696907
Sistemas Inteligentes IV
"""
from turtle import width
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import image 
from matplotlib import pyplot as plt
from tensorflow import keras

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Our operations on the frame come here
    width,height = frame.shape[:2]
    a = width/2
    b = height/2
    print(a, "\n", b)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _ , img_bn = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    binary = cv.resize(img_bn, (28,28))
    patron = binary.reshape(1,28,28,1)/255
    # variable patron lista para la generalización!
    
    #model = keras.models.load_model('CNN_Model_MNIST.h5')
    
    #ypred = model.predict(patron)
    #y_pred = np.argmax(ypred,axis=1)

    #print("Número: ", y_pred, "\n")

    
    # Display the resulting frame
    
    cv.imshow('frame', gray)

    
    if cv.waitKey(1) == ord('q'):
        cv.imwrite("im_1.png",frame)
        cv.imwrite("im1_2.png",img_bn)
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

data = image.imread('im_1.png') 
data1 = np.zeros(data.shape)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if sum(data[i][j]) > 1.1:
            data1[i][j] = np.array([1,1,1])
        else:
            data1[i][j] = np.array([0,0,0])

plt.plot(320, 240  , marker='*', color="red") 
plt.imshow(data1) 
plt.show() 
