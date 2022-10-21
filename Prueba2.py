"""
Created on Sat Oct  1 23:22:55 2022

@author: CMC SCANNER
"""

import cv2
import numpy
import numpy as np
import math
from pyautocad import Autocad, APoint

# --------------- Procesamiento de la imagen ----------------------------------
imagen = cv2.imread('im.png')
imageng = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
[l,a] = numpy.shape(imageng)
filas = []
columnas_inicio = []
columnas_final = []
posiciones = []
inicial = 0
final = 1
con = 0
auxiliar = 0
for i in range(l):
    for j in range(a):
        if(imageng[i,j] != 0):
            if(con != 0):
                final = j
            else:
                inicial = j
                con = 1
    con = 0
    if(numpy.mean(imageng[i,inicial:final]) != 0):
        columnas_inicio.append(inicial)
        columnas_final.append(final)
        filas.append(i)
longitudp = len(filas)
for i in range(longitudp):
    auxiliar = int((((columnas_final[i]-columnas_inicio[i]))/2)+1)
    posiciones.append(columnas_inicio[i]+auxiliar)
    imageng[filas[i],columnas_inicio[i]:columnas_final[i]] = 0
    imageng[filas[i],posiciones[i]] = 255

# ------------------- cambio de coordenadas ----------------------------------
theta = 22.5
phi = 22.5
d = 19
FOV = 88
xx = []
yy = []
zz = []
fc = math.sqrt(l^2+a^2)/(2*math.tan(FOV/2))
for i in range(longitudp):
    d_phi = np.arctan2(math.sqrt(pow(posiciones[i],2)+pow(filas[i],2)),fc)
    zz.append(d*(math.cos(d_phi)*math.cos(theta))/math.sin(phi+theta-d_phi))
    xx.append(zz[i]*(posiciones[i]/fc))
    yy.append(zz[i]*(filas[i]/fc))
datos123 = []
for i in range(longitudp):
    datos123.append([xx[i],yy[i],zz[i]])

# --------------- Parte de Autocad ------------------
# acad = Autocad()
# acad.prompt("Hello, Autocad fron Python\n")     
# print(acad.doc.Name) 
# b = len(datos123)-1
# for i in range(b):
#     p1 = APoint(datos123[i])
#     p2 = APoint(datos123[i+1]) 
#     acad.model.AddLine(p1,p2)