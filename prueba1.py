
import cv2 
import numpy as np 

"""import cv2 as cv
import numpy as np

from tensorflow import keras

cap = cv.VideoCapture(1, cv.CAP_DSHOW)

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
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    _ , img_bn = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    binary = cv.resize(img_bn, (28,28))
    
    patron = binary.reshape(1,28,28,1)/255
    # variable patron lista para la generalización!
    
    model = keras.models.load_model('CNN_Model_MNIST.h5')

    ypred = model.predict(patron)
    y_pred = np.argmax(ypred,axis=1)

    print("Número: ", y_pred, "\n")

    
    # Display the resulting frame
    cv.imshow('frame', img_bn)

    
    if cv.waitKey(1) == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
"""
"""width = int(im1.shape[1] * 75 / 100)
    height = int(im1.shape[0] * 75 / 100)
# dsize
    dsize = (width, height)
# cambiar el tamaño de la image
    im = cv2.resize(im, dsize)
    im1 = cv2.resize(im1, dsize)
    cv2.imshow('.',im1) """
# Definicion del primer ejercicio, con funcion llamada "ejercicio1", recibiendo las imagenes solicitas
def ejercicio1(im): 
    im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(im1, 165, 170, cv2.THRESH_BINARY)
    valH=1         # Valores de los fondos de cada imagen solicitada.
    titu='Binarizada'        # Ciclo en donde se crean la mascara para cada imagen tomando la imagen y los valores de brillos de fondo
    #mask = cv2.inRange(im1,(59,0,0),(61,255,210))
    cv2.imshow(titu,thresh1)        # Se muestra la imagen con la mascara creada
    cv2.imshow('Original',im)
    cv2.waitKey(0)
    width = int(im1.shape[1])
    height = int(im1.shape[0])
    print(width)

def main():
    im = cv2.imread('prueba14.png')
    ejercicio1(im)

if __name__ == '__main__':
    main()
