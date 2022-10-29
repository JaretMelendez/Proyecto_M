# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:23:35 2022

@author: jos-j
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 21:27:24 2022
Proyecto Modular "Escaner Laser"
"""


import cv2
import numpy
import numpy as np
import math
import random
import statistics
from pyautocad import Autocad, APoint
from numpy import random as rd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import matplotlib.pyplot as plt


# --------------- Procesamiento de la imagen --------
imagen = cv2.imread('im.png')
imageng = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
imageng = cv2.medianBlur(imageng,7)
imageng = cv2.GaussianBlur(imageng,(7,7),0)
[l,a] = numpy.shape(imageng)
#cv2.imshow('-',imageng)
#cv2.waitKey(0)

promc = []
vectores = []
vectoresf = []
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
        vectores.append(inicial)
        vectoresf.append(final)
        promc.append(i)
longitudp = len(promc)
for i in range(longitudp):
    auxiliar = vectoresf[i]-vectores[i]
    if((auxiliar%2)!=0):
        auxiliar = (auxiliar-1)/2
        auxiliar = auxiliar + 1
    else:
        auxiliar = auxiliar/2
        auxiliar = auxiliar + 1
    posiciones.append(vectores[i]+int(auxiliar))
    imageng[promc[i],vectores[i]:vectoresf[i]] = 0
for i in range(longitudp):
    imageng[promc[i],posiciones[i]] = 255

# ------------------- cambio de coordenadas 
acumulador = 0
theta = 0
phi = (25*math.pi)/180
d = 190
FOV = (88*math.pi)/180
step = 180
gtmotor = (4*math.pi)/180
xx = []
yy = []
zz = []
zz1 = []
xx1 = []
yy1 = []
datos123 = []
for j in range(step):
    if( j<= 90):
        katmotor = j*gtmotor
    else:
        katmotor = -j*gtmotor
    fc = math.sqrt(pow(l,2)+pow(a,2))/(2*math.tan(FOV/2))
    for i in range(longitudp):
        acumulador = pow(posiciones[i],2)+pow(promc[i],2)
        d_phi = np.arctan2(math.sqrt(acumulador),fc)
        k = phi + theta - d_phi
        zz.append(d*(math.cos(d_phi)*math.cos(theta))/math.sin(k)+math.cos(katmotor))
        xx.append((zz[i]*(posiciones[i]/fc)))
        yy.append((zz[i]*(promc[i]/fc)))
        phi = k - theta + d_phi
        
    print(j)
    
for i in range(len(xx)):
    datos123.append([xx[i],yy[i],zz[i]])
    """
    for i in range(longitudp):
        xx1.append([xx[i]])
        yy1.append([yy[i]])
        zz1.append([zz[i]])"""
    ###############################################################################
    ##############################################################################    
    
    # Nuestra entrada normalizada 
    
    """
    ##############################################################################
    ##############################################################################    
      
    # data_arr = [xx,yy,zz]
    # all_indices = list(range(xx)[0:len(xx)])
    # train_ind, test_ind = train_test_split(all_indices, test_size=0.2)
    # train = data_arr[:,:,train_ind,:]
    # test = data_arr[:,:,test_ind, :]
    
    for i in range(len(xx)):
        datos123.append([xx[i],yy[i],zz[i]])"""
    # --------------- Parte de Autocad ------------------
acad = Autocad()
    #acad = Autocad(create_if_not_exists=True)    
    
b = len(datos123)-1
for i in range(b):
    p1 = APoint(datos123[i])
    p2 = APoint(datos123[i+1]) 
    acad.model.AddLine(p1,p2)
print(acad.doc.Name) 