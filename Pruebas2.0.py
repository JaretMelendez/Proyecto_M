import cv2
import numpy
import numpy as np
import math
import random
import statistics
import matplotlib.pyplot as plt

from pyautocad import Autocad, APoint
from numpy import random as rd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec

# --------------- Motor ------------------


# --------------- Toma de muestras ------------------
def Camara():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    step = 1
    i=0
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
        cv2.imwrite("img"+str(i+6)+".png",frame)
        if i == step-1:
            break
        i= i+1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# --------------- Procesamiento de la imagen --------
def Mascara(imagen):
    imageng = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    [l,a] = numpy.shape(imageng)
    final = 0
    newi2 = numpy.zeros((l,a),dtype=numpy.uint8)
    for i in range(l):
        for j in range(a):
            if(imageng[i,j] >= 30):
                final = j
        if(numpy.mean(imageng[i,final]) != 0):
            newi2[i,final] = 255
        final = 0
    return newi2

# --------------- Coordenadas ------------------

# --------------- Conexion Autocad ------------------
def Autocad(datos123):
    acad = Autocad(create_if_not_exists=True)
    print(acad.doc.Name)
    acad.prompt("Hello, Autocad fron Python\n")     
    b = len(datos123)-1
    for i in range(b):
        p1 = APoint(datos123[i])
        p2 = APoint(datos123[i+1]) 
        line = acad.model.AddLine(p1,p2)
        line.Color = 10 
# --------------- Funcion Principal ------------------
def main():
    imagen = cv2.imread('im.png')
    Imagen = Mascara(imagen)
    Autocad()

if __name__ == "__main__":
    main()