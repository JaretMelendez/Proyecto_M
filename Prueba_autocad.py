# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 21:27:24 2022
Proyecto Modular "Escaner Laser"
@author: jaret
"""
import cv2
import numpy
import numpy as np
from matplotlib import image 
from matplotlib import pyplot as plt
import math
# --------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pickle
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

# -------------- Librerias para Autocad -------------
#import pyautocad
from pyautocad import Autocad, APoint
import pandas as pd

# ---------------------------------------------------
#data = image.imread('im.png')
imagen = cv2.imread('im.png')
b,g,r = cv2.split(imagen)
plt.figure()
plt.imshow(imagen)
[l,a] = numpy.shape(b)
aa = numpy.array([l,a])
for i in range(l):
    for x in range(a):
        if(b[i][x] != 0):
            cb = 0
            if(b[i][x] < 0):
                b[i][x] = 0
            if((b[i][x]+cb)>=255):
                b[i][x] = 255
        if(r[i][x] != 0):
            cr = 0
            if((r[i][x]+cr)>=255):
                r[i][x] = 255
        if(g[i][x] != 0):
            cg = 255
            if((g[i][x]+cg)>=255):
                g[i][x] = 255
for i in range(l):
    for x in range(a):
        if(b[i][x] != 0):
            cb = b[i][x]
            if(cb>255):
                b[i][x] = 255
            elif(cb < 60):
                b[i][x] = 0
        if(r[i][x] != 0):
            cr = r[i][x]
            if(cr>255):
                r[i][x] = 255
            elif(cr < 32):
                r[i][x] = 0
        if(g[i][x] != 0):
            cg = b[i][x]
            if(cg>255):
                g[i][x] = 255
            elif(cg < 230):
                g[i][x] = 0
imagenn = cv2.merge((r,g,b))
plt.figure()
plt.imshow(imagenn)
data = imagenn 
data1 = np.zeros(data.shape)
# ---------------------------------------------------
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if sum(data[i][j]) > 0.7:
            data1[i][j] = np.array([1,1,1])
        else:
            data1[i][j] = np.array([0,0,0])
plt.figure()
plt.imshow(data1)
h = data1.shape[0]
w = data1.shape[1]
a = np.empty((0,2),int)

for i in range(w):
    for j in range(h):
        if data1[j,i].any() != 0:
            a = np.append(a, np.array([[j,i]]),axis=0)
# read frame2, resize and convert to grayscale
theta = 22.5
phi = 22.5
d = 19
FOV = 88
u = a[:,0]
v = a[:,1]
fc = np.sqrt(w^2+h^2)/(2*np.tan(FOV/2))
d_phi = np.arctan2(np.sqrt(u**2+v**2),fc)
z = d*(np.cos(d_phi)*math.cos(theta))/np.sin(phi+theta-d_phi)
x = z*(u/fc)
y = z*(v/fc)

data = {'X': x, 'Y': y, 'Z': z}
df = pd.DataFrame(data)
df.to_csv('dfpp.csv',index=False)

# -----------------------------------
data = pd.read_csv("dfpp.csv")
df = pd.DataFrame(data)
x = data['X']
y = data['Y']
z = data['Z']
x1 = np.empty((0,1),int)
y1 = np.empty((0,1),int)
z1 = np.empty((0,1),int)
for i in x[1::6]:
    x1 = np.append(x1,i)
for i in y[1::6]:
    y1 = np.append(y1,i)
for i in z[1::6]:
    z1 = np.append(z1,i)

# ---------------------------------------------------
x = x1
y = y1
z = z1
a = len(x)-1
xa =[]
ya =[]
za =[]
xa1 =[]
ya1 =[]
za1 =[]
suma = []
suma2 = []
datos = []
datos123 = []
ax = 0
ay = 0
az = 0
i = 0
print(len(x))
for j in range(len(x)):
    suma.append(abs(y[j]+x[j]+z[j]))
suma.sort()
for w in range(len(x)):
    for r in range(len(x)):
        sumac = (abs(y[r]+x[r]+z[r]))
        if ((suma[w] == sumac) and (i!=1)):
            datos.append([y[r],x[r],z[r]])
            xa.append(y[r])
            ya.append(x[r])
            za.append(z[r])
            i=1
    i=0
        
print(w)
print(len(datos))

for j in range(len(xa)):
    suma2.append(abs(ya[j]+xa[j]+za[j]))
suma2.sort()
con = len(xa)-1
for f in range(con):
    sumac = (abs(ya[f+1]+xa[f+1]+za[f+1]))
    if(abs(suma2[f]-abs(ya[f+1]+xa[f+1]+za[f+1]))<0.01):
        ax = (ax + xa[f+1])
        ay = (ay + ya[f+1])
        az = (az + za[f+1])
        i = i+1
    else:
        if(i != 0):
            datos123.append([ax/i,ay/i,az/i])
            xa1.append(ax/i)
            ya1.append(ay/i)
            za1.append(az/i)
        else:
            datos123.append([xa[f+1],ya[f+1],za[f+1]])
            xa1.append(xa[f+1])
            ya1.append(ya[f+1])
            za1.append(za[f+1])
        ax = 0
        ay = 0
        az = 0
        i=0

# ---------------------------------------------------
'''
x11 = np.transpose(np.asanyarray([xa1]))
y11 = np.transpose(np.asanyarray([ya1]))
x_train, x_test, y_train, y_test = train_test_split(x11,y11)

plt.figure()
plt.grid()
plt.title('Regresion no lineal')
plt.xlabel('dis')
plt.ylabel('vol')

plt.plot(x_train,y_train,'bo')
plt.plot(x_test,y_test,'ro')
plt.legend(['Entrenamiento', 'Generalización'])
"""
model = Pipeline([('scaler',StandardScaler()),
                  ('regresion',KernelRidge(alpha=0.4,kernel='rbf'))])
"""
model = Pipeline([('scaler',StandardScaler()),
                  ('regresion',SVR(epsilon=0.2,C=200,kernel=('rbf')))])


model.fit(x_train,y_train.ravel())

print('Train score: ',model.score(x_train,y_train))
print('Test score: ',model.score(x_test,y_test))

x_plot = np.linspace(x.min(), x.max(),50).reshape(-1,1)
y_plot = model.predict(x_plot)

m = []
num = len(x_plot)

for i in range(num-1):
    m.append((y_plot[i]-y_plot[i+1])/(x_plot[i]-x_plot[i+1]))
    m.append((y_plot[num-2]-y_plot[num-1])/(x_plot[num-2]-x_plot[num-1])) 

fig=plt.figure(figsize=(8,6))
axes = plt.axes(projection="3d")
#axes.scatter3D(x_plot,y_plot,m)
axes.set_title("3d Line plot in Matplotlib",fontsize=14,fontweight="bold")
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_zlabel("Z")  
plt.tight_layout()
plt.show() 

datos123 = []
for i in range(num):
    datos123.append([y_plot[i],x_plot[i],m[i]])'''
       
# --------------- Parte de Autocad ------------------
acad = Autocad()
#acad = Autocad(create_if_not_exists=True)
acad.prompt("Hello, Autocad fron Python\n")     
print(acad.doc.Name)
print(len(suma))
print(len(datos))
print(len(x))    
b = len(datos123)-1
for i in range(b):
    p1 = APoint(datos123[i])
    p2 = APoint(datos123[i+1]) 
    acad.model.AddLine(p1,p2)

