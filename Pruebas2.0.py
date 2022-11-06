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

# --------------- Procesamiento de la imagen --------
imagen = cv2.imread('im.png')
imageng = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
[l,a] = numpy.shape(imageng)
# Variante 1 - Proceso manual
final = 0
newi2 = numpy.zeros((l,a),dtype=numpy.uint8)
for i in range(l):
    for j in range(a):
        if(imageng[i,j] >= 30):
            final = j
    if(numpy.mean(imageng[i,final]) != 0):
        newi2[i,final] = 255
    final = 0
cv2.imshow('Original',imagen)
cv2.imshow('Manual',newi2)
cv2.waitKey(0)
"""
# ------------------- cambio de coordenadas 
acumulador = 0
theta = 22.5
phi = 22.5
d = 19
FOV = 88
xx = []
yy = []
zz = []
fc = math.sqrt(l^2+a^2)/(2*math.tan(FOV/2))
for i in range(longitudp):
    acumulador = pow(vectoresf[i],2)+pow(promc[i],2)
    d_phi = np.arctan2(math.sqrt(acumulador),fc)
    zz.append(d*(math.cos(d_phi)*math.cos(theta))/math.sin(phi+theta-d_phi))
    xx.append(zz[i]*(vectoresf[i]/fc))
    yy.append(zz[i]*(promc[i]/fc))
datos123 = []
for i in range(len(xx)):
    datos123.append([xx[i],yy[i],zz[i]])

def Histograma(imagen,canales,mascara,tamaño,rangos,constante):
    img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    img2 = cv2.add(img,constante)
    a=cv2.calcHist([img] ,canales ,mascara ,tamaño ,rangos)
    a1=cv2.calcHist([img2] ,canales ,mascara ,tamaño ,rangos)
    x= range(256)
    ax1 = plt.subplot(211)
    ax1.bar(x, a.reshape(256))
    ax1.set_title('Imagen escala de Grises')
    ax2 = plt.subplot(212)
    ax2.bar(x, a1.reshape(256))
    ax2.set_title('Imagen escala de Grises con brillo')
    plt.show()
    cv2.imshow('Imagen original escala de Grises',img)
    cv2.imshow('Imagen original escala de Grises con brillo',img2)
    cv2.waitKey(0)

Histograma(imagen,[0],None,[256],[0,256],200)

"""
# --------------- Parte de Autocad ------------------
#acad = Autocad()
'''
acad = Autocad(create_if_not_exists=True)
print(acad.doc.Name)
acad.prompt("Hello, Autocad fron Python\n")     
b = len(datos123)-1
for i in range(b):
    p1 = APoint(datos123[i])
    p2 = APoint(datos123[i+1]) 
    line = acad.model.AddLine(p1,p2)
    line.Color = 10 '''