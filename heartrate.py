import numpy as np
import matplotlib.pyplot as plt
import cv2
import lib_fourier


def byPartitions(filename, sector=[1, 1]):
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("No lo pude abrir")
        exit(-1)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    partX=height//3
    partY=width//3
    posX = sector[0] * partX
    posY = sector[1] * partY
    piX = partX
    piY = partY

    return process(filename,posX,posY,piX,piY)

def bySize(filename, size=30):
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("No lo pude abrir")
        exit(-1)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    sizeX=min(height,size)
    sizeY=min(width,size)

    posX = height//2 - sizeX//2
    posY = width//2 - sizeY//2
    piX = sizeX
    piY = sizeY

    return process(filename,posX,posY,piX,piY)

def process(filename, posX,posY,piX,piY):
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("No lo pude abrir")
        exit(-1)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    r = np.zeros((1, length))
    g = np.zeros((1, length))
    b = np.zeros((1, length))

    k = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        r[0, k] = np.mean(frame[posX:posX + piX, posY:posY + piY, 0])
        g[0, k] = np.mean(frame[posX:posX + piX, posY:posY + piY, 1])
        b[0, k] = np.mean(frame[posX:posX + piX, posY:posY + piY, 2])
        k = k + 1

    n = 2 ** int(np.log2(k))

    cap.release()

    f = np.linspace(-n / 2, n / 2 - 1, n) * fps / n

    r = r[0, 0:n] - np.mean(r[0, 0:n])
    g = g[0, 0:n] - np.mean(g[0, 0:n])
    b = b[0, 0:n] - np.mean(b[0, 0:n])

    R = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(r))) ** 2
    G = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(g))) ** 2
    B = np.abs(np.fft.fftshift(lib_fourier.cooley_fft(b))) ** 2

    return f, R, G, B, r,g,b



if (__name__ == "__main__"):
    archivo="ota.mp4"

    for x in range(3):
        for y in range(3):
            f,R,G,B,r,g,b=byPartitions(archivo,[x,y])
            plt.plot(60*f,  B)
            print("Frecuencia cardíaca: ", abs(f[np.argmax(B)]) * 60, "AZUL ppm en",[x,y])
    plt.xlim(0, 200)
    plt.xlabel("cuadro del video")
    plt.show()

    for x in [50,500,1000,2000]:
        f, R, G, B, r, g, b=bySize(archivo, x)
        print("Frecuencia cardíaca: ", abs(f[np.argmax(B)]) * 60, "AZUL ppm")
        plt.plot(60 * f, B)
    plt.xlim(0, 200)
    plt.show()

