
#from ctypes import sizeof
#from turtle import width
import cv2 as cv
import numpy as np
import pandas as pd
import pyautocad
#from PIL import Image
from matplotlib import image 
from matplotlib import pyplot as plt
from tensorflow import keras
#from mpl_toolkits import mplot3d
from pyautocad import Autocad, APoint

#import imutils
import math
#from pyautocad import Autocad

"""data1 = np.zeros(data.shape)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if sum(data[i][j]) > 1.8:
            data1[i][j] = np.array([1,1,1])
        else:
            data1[i][j] = np.array([0,0,0])
            
data = image.imread('im.png') 
gray1 = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
h,w = gray1.shape[:2]

m = np.reshape(gray1,[1,w*h])

mean = m.sum()/(w*h)
#print('mean',mean)

ret,binary = cv.threshold(gray1,mean,130,cv.THRESH_BINARY)

# cv.imshow('original',data)
cv.imshow('binary',binary)
cv.waitKey(0) 
"""
#binary = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 1],[1, 0, 1, 0, 0, 1, 0, 1, 0]])
data = image.imread('im_1.png') 
data1 = np.zeros(data.shape)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if sum(data[i][j]) > 1.1:
            data1[i][j] = np.array([1,1,1])
        else:
            data1[i][j] = np.array([0,0,0])
#plt.imshow(data1) 
#plt.show() 
print("-",data1.shape)
h = data1.shape[0]
w = data1.shape[1]
#m = np.reshape(data1,[1,w*h])
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
# u = a.shape[:1]
# v = a.shape[:2]
u = a[:,0]
v = a[:,1]
fc = np.sqrt(w^2+h^2)/(2*np.tan(FOV/2))
d_phi = np.arctan2(np.sqrt(u**2+v**2),fc)
z = d*(np.cos(d_phi)*math.cos(theta))/np.sin(phi+theta-d_phi)
x = z*(u/fc)
y = z*(v/fc)
#print("x: ",len(x))
#print("y: ",len(y))
#print("z: ",len(z))

data = {'X': x, 'Y': y, 'Z': z}
df = pd.DataFrame(data)
df.to_csv('df_color_.csv',index=False)

# Creamos la figura
fig=plt.figure(figsize=(8,6))
axes = plt.axes(projection="3d")
axes.scatter3D(x,y,z)
axes.set_title("3d Line plot in Matplotlib",fontsize=14,fontweight="bold")
axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_zlabel("Z")  
#plt.tight_layout()
#plt.show()
#acad = Autocad()
#point1 = APoint(x,y)
#circle1 = acad.model.AddCircle(point1,z)

file = pd.read_csv("df_color_.csv")

df = pd.DataFrame(file)

x = df['X']
y = df['Y']
z = df['Z']

listx = []
listy = []
listz = []

for i in x:
    listx.append(i)
for j in y:
    listy.append(j)
for k in z:
    listz.append(k)

union = zip(listx,listy,listz)
coordenadas = list(union)

acad = pyautocad.Autocad()

def trazo(p,r):
    acad.model.AddCircle(p,r)
def punto(c):
    acad.model.AddPoint(c)

for q in coordenadas:
    puntos = APoint(q)
    trazo(puntos,2)
    punto(puntos)

acad.prompt("hola")
print("Ya se acabo")