import cv2
import numpy
import numpy as np
import math
import random
import statistics
import matplotlib.pyplot as plt
import time
from pyfirmata import Arduino, util
from pyautocad import Autocad, APoint
from numpy import random as rd, np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec

# --------------- Giro de Motor ------------------
def girar_derecha(DIR,PUL,velocidad,arduino):
    arduino.digital[DIR].write(0)
    arduino.digital[PUL].write(1)
    time.sleep(velocidad)
    arduino.digital[PUL].write(0)
    time.sleep(velocidad)

# --------------- Motor ------------------
def Motor(muestras,acumulador,Puerto_serial):
    DIR = 2 
    PUL = 3
    velocidad = 0.5
    pulsos = 30
    i = 0
    arduino = Arduino(Puerto_serial)
    time.sleep(2)
    for i in range(pulsos):
        girar_derecha(DIR,PUL,velocidad,arduino)
    acumulador = acumulador + i
    if(acumulador == pulsos):
        arduino.digital[PUL].write(0)
        arduino.close()
    return acumulador

# --------------- Toma de muestras ------------------
def Camara(muestra):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return 0
    cv2.imwrite("Muestra"+str(muestra)+".png",frame)


# --------------- Procesamiento de la imagen --------
def Mascara(imagen):
    Imagen_Gray = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    [l,a] = numpy.shape(Imagen_Gray)
    Posicion_final = 0
    longitud_pixeles = 0
    vector_posiciones = []
    Posiciones_columnas = []
    Nueva_imagen = numpy.zeros((l,a),dtype=numpy.uint8)
    for Columnas in range(l):
        for Filas in range(a):
            if(Imagen_Gray[Columnas,Filas] >= 30):
                Posicion_final = Filas
        if(numpy.mean(Imagen_Gray[Columnas,Posicion_final]) != 0):
            Nueva_imagen[Columnas,Posicion_final] = 255
            longitud_pixeles = longitud_pixeles + 1
            vector_posiciones.append(Posicion_final)
            Posiciones_columnas.append(Columnas)
        Posicion_final = 0
    Coordenadas(Nueva_imagen,longitud_pixeles,Posiciones_columnas,vector_posiciones)

# --------------- Coordenadas ------------------
def Coordenadas(Imagen,longitud_pixeles,Posiciones_columnas,vector_posiciones):
    FOV = 845
    Co = 70
    b = 195
    Z0 = 400
    Z1 = 250

# ---------------------------------------------------    
    Autocad()

# --------------- Conexion Autocad ------------------
def Autocad(coordenadas):
    acad = Autocad(create_if_not_exists=True)
    print(acad.doc.Name)
    acad.prompt("Hello, Autocad fron Python\n")     
    tamaño = len(coordenadas)-1
    for contador in range(tamaño):
        p1 = APoint(coordenadas[contador])
        p2 = APoint(coordenadas[contador+1]) 
        line = acad.model.AddLine(p1,p2)
        line.Color = 10 

# --------------- Funcion Principal ------------------
def main():
    acumulador_motor = 0
    Muestras = 92
    Puerto_serial = ("COM4")
    for i in range(Muestras):
        if(acumulador_motor != 0):
            acumulador_motor = retorno_motor
        retorno_motor = Motor(Muestras,acumulador_motor,Puerto_serial)
        Camara(i)
    for j in range(Muestras):
        imagen = cv2.imread("Muestra"+str(j)+".png")
        Mascara(imagen)

if __name__ == "__main__":
    main()