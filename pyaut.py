from pyautocad import Autocad, APoint
from math import sin, cos, pi

autocad = Autocad()
autocad.prompt("Hola mundo")
long = 1 
angulo0 = 1
angulo1 = 2.1
spaciado = 1
x = [0]
y = [0]

for i in range(0, 10000):
    p0 = APoint(x[-1], y[-1])
    x.append(x[-1] + long*sin(angulo0*i*pi/180))
    y.append(y[-1] + long*cos(angulo0*i*pi/180))
    p1 = APoint(x[-1],y[-1])
    autocad.model.AddLine(p0,p1)

    x.append(x[-1] + long*cos(angulo1*i*pi/180))
    y.append(y[-1] + long*sin(angulo1*i*pi/180))

    autocad.model.AddLine(p1,p0)