import numpy as np
import matplotlib.pyplot as plt
import cv2
import lib_fourier
import json
from scipy.signal import butter, filtfilt

with open('config.json') as f:
            config = json.load(f)

cap = cv2.VideoCapture(config["file"])

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
gr = np.zeros((1,length))

posX=int(config["posX"])
posY=int(config["posY"])
pix=int(config["size"])

k = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break
    if posX>np.shape(frame)[0]:
        raise Exception("invalid posX")
    if posX > np.shape(frame)[1]:
        raise Exception("invalid posY")

    gris=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r[0,k] = np.mean(frame[posX:posX+pix,posY:posY+pix,2])
    g[0,k] = np.mean(frame[posX:posX+pix,posY:posY+pix,1])
    b[0,k] = np.mean(frame[posX:posX+pix,posY:posY+pix,0])
    gr[0,k] = np.mean(gris[posX:posX+pix,posY:posY+pix])

    k = k + 1

n=2**int(np.log2(k))


#n = 1024
f = np.linspace(-n/2,n/2-1,n)*fps/n

r = r[0,0:n]-np.mean(r[0,0:n])
g = g[0,0:n]-np.mean(g[0,0:n])
b = b[0,0:n]-np.mean(b[0,0:n])
gr = gr[0,0:n]-np.mean(gr[0,0:n])


minBpm, maxBpm = 40, 220
if config["filter"]:
    frec = [2 * (minBpm / 60) / fps, 2 * (maxBpm / 60) / fps]
    b_, a_ = butter(2, frec, btype='bandpass')
    r = filtfilt(b_, a_, r)
    g = filtfilt(b_, a_, g)
    b = filtfilt(b_, a_, b)
    gr = filtfilt(b_,a_,gr)

R = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(r)))**2
G = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(g)))**2
B = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(b)))**2
GR = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(gr)))**2

plt.plot(60*f,R,'r')
plt.xlim(0,200)

plt.plot(60*f,G,'g')
plt.xlim(0,200)
plt.xlabel("frecuencia [1/minuto]")

plt.plot(60*f,B,'b')
plt.xlim(0,200)


plt.plot(60*f,GR,'y')
plt.xlim(0,200)



plt.show()

print("Frecuencia cardíaca: ", abs(f[np.argmax(R)])*60, "red pulsaciones por minuto")
print("Frecuencia cardíaca: ", abs(f[np.argmax(G)])*60, "green pulsaciones por minuto")
print("Frecuencia cardíaca: ", abs(f[np.argmax(B)])*60, "blue pulsaciones por minuto")
print("Frecuencia cardíaca: ", abs(f[np.argmax(GR)])*60, "gray pulsaciones por minuto")

