import cv2 
import numpy as np 
from matplotlib import pyplot as plt

# Definicion de la función umbralizacion recibiendo los parametros imagen, valorUmbral, valorMaximo, tecnicaUmbralizacion
def umbralizacion(imagen, valorUmbral, valorMaximo, tecnicaUmbralizacion):
    im=imagen           #Asigno la imagen original en grises a la variable "im"
    lim = np.arange(6)  # Lim es inicializado como una variable con un valor de 6
    ret, thresh1 = cv2.threshold(im, valorUmbral, valorMaximo, tecnicaUmbralizacion[0])
    ret, thresh2 = cv2.threshold(im, valorUmbral, valorMaximo, tecnicaUmbralizacion[1])
    ret, thresh3 = cv2.threshold(im, valorUmbral, valorMaximo, tecnicaUmbralizacion[2])
    ret, thresh4 = cv2.threshold(im, valorUmbral, valorMaximo, tecnicaUmbralizacion[3])
    ret, thresh5 = cv2.threshold(im, valorUmbral, valorMaximo, tecnicaUmbralizacion[4])
    # cv2.threshold realiza el proceso de umbralizacion con cada una de las tecnicas de umbralización y es 
    # asignado en distintas variables para casa tecnica.
    imag=[im, thresh1, thresh2, thresh3, thresh4, thresh5]
    # "imag" almacena un todos los procesos de umbralizacion en un arreglo de 6 valores
    return imag     # Retorna todas la imagenes con cada tecnica 
    
def main():
#cv2.THRESH_BINARY  #cv2.THRESH_BINARY_INV  #cv2.THRESH_TRUNC  #cv2.THRESH_TOZERO #cv2.THRESH_TOZERO_INV
    im = cv2.imread('HaloRe.jpg')              
    img= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)       #Conversión de la imagen original a grises 
    tecumbra = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV,       #Asignacion de las tecnicas
                cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
    imag=umbralizacion(img, 100, 190, tecumbra) #LLamada a la funcion 
    titles = ['Original a Grises', 'BINARY',
              'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    lim = np.arange(6)
    for i in lim:
        plt.subplot(2, 3, i+1), plt.imshow(imag[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()

