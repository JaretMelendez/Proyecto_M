# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 23:22:55 2022

@author: jaret
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
[l,a] = numpy.shape(imageng)
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
theta = 22.5
phi = 22.5
d = 19
FOV = 88
xx = []
yy = []
zz = []
fc = math.sqrt(l^2+a^2)/(2*math.tan(FOV/2))
for i in range(longitudp):
    acumulador = pow(posiciones[i],2)+pow(promc[i],2)
    d_phi = np.arctan2(math.sqrt(acumulador),fc)
    zz.append(d*(math.cos(d_phi)*math.cos(theta))/math.sin(phi+theta-d_phi))
    xx.append(zz[i]*(posiciones[i]/fc))
    yy.append(zz[i]*(promc[i]/fc))
datos123 = []
for i in range(len(xx)):
    datos123.append([xx[i],yy[i],zz[i]])
zz1 = []
xx1 = []
yy1 = []
for i in range(longitudp):
    xx1.append([xx[i]])
    yy1.append([yy[i]])
    zz1.append([zz[i]])
###############################################################################
##############################################################################    


## Funciones 
def logistica(alpha,v):
    y = 1/(1+math.exp(- np.dot(alpha,v)))
    return y


def dlogistica(alpha,y):
    dy = np.dot(alpha , np.dot(y , (1-y)))
    return dy


def standardScaler(V):
    N = (V - np.mean(V))/np.std(V);
    return N


# Des Normalizar datos
# D -> Vector Desnormalizado
# N -> Vector normalizado
# V -> Vectror Original
def standardScalerInv(N,V):
    D = N * np.std(V) + np.mean(V)
    return D


def gradientanh(y):
    g = np.dot((1 - y) , (1+y))
    return g



## Prepara los datos para la entrada a la red 
[l1,l2]= numpy.shape(yy1)
tam = l1*l2;
ss = StandardScaler()
xn = ss.fit_transform(xx1)
yn = ss.fit_transform(yy1)
dn = ss.fit_transform(zz1)

x1 = numpy.reshape(xn,(1,tam))
y1 = numpy.reshape(yn,(1,tam))
z1 = numpy.reshape(dn,(1,tam))

# Organizamos los datos en arreglos

x1 = list(x1)
y1 = list(y1)
# Nuestra entrada normalizada 
inputN = [x1,y1]   
# Grafica datos normalizados 
## Seleccionar datos de entrenamiento y datos para generalización
indice = np.random.permutation(tam)
tamentre = math.floor(tam * 0.8)
indientre = indice[1:tamentre]
indigenera = indice[tamentre+1:tam]

# Declarar la red  
# 2 inputs, 5 neu c_oculta
n_ent = 2
n_neuronas_1 = 5   
n_neuronas_2 = 1

# Vectores de pesos
# Pesos iniciados de manera aleatoria
W_1 = numpy.random.rand(n_neuronas_1, n_ent+1)
W_2 = numpy.random.rand(n_neuronas_2, n_neuronas_1+1)
l = (n_ent+1)*n_neuronas_1+(n_neuronas_1+1)*n_neuronas_2

# Parametros de entrenamiento 
alpha = 0.01
eta = 0.00001
P = numpy.eye(l,l)*10000
Q = numpy.eye(l,l)*10000
R = numpy.eye(n_neuronas_2,n_neuronas_2)*1000
H = numpy.zeros((l,n_neuronas_2))
epocas = 100

for i in range(epocas):
    mezclar_indice = np.random.permutation(tamentre)
    mezclar_indice = list(mezclar_indice)
    er = 0;
    for j in range (tamentre):
        # calcular salida capa 1
        inputt=[1,inputN[:,indientre[mezclar_indice[j]]]]
        v_1 = W_1*inputt
        y_1 = logistica(v_1,alpha)
        inputt2 = [1,y_1]
        v = W_2*inputt2  
        e = dn[indientre[mezclar_indice[j]]]-y
        [tw1,tw2] = numpy.shape(W_1)
        WK = [numpy.reshape(W_1,tw1*tw2),W_2]
        # Para mas capas y mas neuronas por capa seguir el ejemplo de
        # arriba 
        #Funcion de activacion lineal
        Y = np.dot(W_2,Y_1)
        
        #Calcular el error 
        e = dn[indientre[i]]-Y
        er = er + math.pow(e,2)
        # Calcular gradiente global
        # Para función lineal es e*1
        grad_glob = e
        
        # Calcular gradiente local 
        for j in range(n_neuronas_1):
            dY_1[j] = dlogistica(alpha,Y_1[j,0]) #derivadas
            grad_loc_1[j,0] = dY_1[j]*(W_2[0,j]*grad_glob)
        
        # Actualizar pesos 
        for j in range(n_neuronas_1):
           W_1[j,:] = W_1[j,:] +(eta* input2*grad_loc_1[j])
        
        W_2 = W_2 + (eta * grad_glob * Y_1.transpose())
        
    err[epoca] = math.sqrt(er/tamentre)

## Grafica error 
tiempo= range(epocas)

Zn = np.zeros((tam,1))

## Probamos con todos los datos 
for i in range(tam-1): 
    # Calcular las salidas de capa 1 -9+
    inputt[0][0] = inputN[0,i]
    inputt[0][1] = inputN[1,i]
    input2 = np.insert(inputt,2,1,axis=1)
    for j in range(n_neuronas_1):
        Y_1[j,0]= logistica(alpha,np.dot(input2,W_1[j,:]))
    
    # Para mas capas y mas neurona se usa el mismo algoritmo
    Y = np.dot(W_2,Y_1)
    Zn[i]=Y

# Usamos reshape para ajusta las dimensiones 
## Graficas finales 
# Grafica datos normalizados 
Z_hat = np.zeros((len(xx),1))
X_1n = np.zeros((len(xx),1))
Y_1n = np.zeros((len(xx),1))
D_1n = np.zeros((len(xx),1))
for i in range(l2):
    for j in range(1):
        Z_hat[i] = Zn[(i)*j]
        X_1n[i] = xn[(i)*j]
        Y_1n[i] = yn[(i)*j]
        D_1n[i] = dn[(i)*j]
        
        
#Z_hat = np.reshape(Zn,[int(l1),int(l1)])
#X_1n = Xn.reshape(l1,l2,order='F').copy()
#Y_1n = Yn.reshape(l1,l2,order='F').copy()
#D_1n = dn.reshape(l1,l2,order='F').copy()

## Analisis de datos de generalizacion 
for i in range(tamentre+1,tam): 
    indiGen = i-tamentre
    # Calcular las salidas de capa 1 
    inputt[0][0] = inputN[0,i]
    inputt[0][1] = inputN[1,i]
    input2 = np.insert(inputt,2,1,axis=1)
    for j in range(n_neuronas_1):
        Y_1[j,0]= logistica(alpha,np.dot(input2,W_1[j,:]))
    
    # Para mas capas y mas neurona se usa el mismo algoritmo
    Y = np.dot(W_2,Y_1)
    
    e = dn[indigenera[indiGen-1]]-Y
    er = er+math.pow(e,2)
    
errG = math.sqrt(er/(tam-tamentre))

print('Media del error de la generalización = %f \n',errG)

    
    
##############################################################################
##############################################################################    
  
# data_arr = [xx,yy,zz]
# all_indices = list(range(xx)[0:len(xx)])
# train_ind, test_ind = train_test_split(all_indices, test_size=0.2)
# train = data_arr[:,:,train_ind,:]
# test = data_arr[:,:,test_ind, :]



for i in range(len(X_1n)):
    datos123.append([X_1n[i],Y_1n[i],Z_hat[i]])
# --------------- Parte de Autocad ------------------
acad = Autocad()
#acad = Autocad(create_if_not_exists=True)
acad.prompt("Hello, Autocad fron Python\n")     
print(acad.doc.Name) 
b = len(datos123)-1
for i in range(b):
    p1 = APoint(datos123[i])
    p2 = APoint(datos123[i+1]) 
    acad.model.AddLine(p1,p2)