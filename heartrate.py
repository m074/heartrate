import numpy as np
import matplotlib.pyplot as plt
import cv2
import lib_fourier

cap = cv2.VideoCapture('ota.mp4')

if not cap.isOpened():
   print("No lo pude abrir")
   exit(-1)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

r = np.zeros((1,length))
g = np.zeros((1,length))
b = np.zeros((1,length))

posX=500
posY=500
pix=30

k = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break
    r[0,k] = np.mean(frame[posX:posX+pix,posY:posY+pix,0])
    g[0,k] = np.mean(frame[posX:posX+pix,posY:posY+pix,1])
    b[0,k] = np.mean(frame[posX:posX+pix,posY:posY+pix,2])
    k = k + 1

n=2**int(np.log2(k))



cap.release()

#n = 1024
f = np.linspace(-n/2,n/2-1,n)*fps/n

r = r[0,0:n]-np.mean(r[0,0:n])
g = g[0,0:n]-np.mean(g[0,0:n])
b = b[0,0:n]-np.mean(b[0,0:n])
#

# R = np.abs(np.fft.fftshift(np.fft.fft(r)))**2
# G = np.abs(np.fft.fftshift(np.fft.fft(g)))**2
# B = np.abs(np.fft.fftshift(np.fft.fft(b)))**2


R = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(r)))**2
G = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(g)))**2
B = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(b)))**2



plt.plot(60*f,R)
plt.xlim(0,200)


plt.plot(60*f,G)
plt.xlim(0,200)
plt.xlabel("frecuencia [1/minuto]")

plt.plot(60*f,B)
plt.xlim(0,200)
plt.show()

print("Frecuencia cardíaca: ", abs(f[np.argmax(R)])*60, "red pulsaciones por minuto")
print("Frecuencia cardíaca: ", abs(f[np.argmax(G)])*60, "green pulsaciones por minuto")
print("Frecuencia cardíaca: ", abs(f[np.argmax(B)])*60, "blue pulsaciones por minuto")