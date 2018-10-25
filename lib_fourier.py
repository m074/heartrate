import numpy as np
import time

def discrete_fourier_transform(serie):
    N=len(serie)
    ans = np.empty((N,)).astype(complex)
    for x in range(N):
        cache=0j
        for y in range(N):
            cache+=serie[y]*np.exp((-2*np.pi*1j*x*y)/N)
        ans[x]=cache
    return ans

def __fft_rec(serie,N):
    ans = np.empty((N,)).astype(complex)
    if N==1 :
        ans[0]=serie[0]
    else:
        left=__fft_rec(serie[::2],N//2)
        right=__fft_rec(serie[1::2],N//2)
        for k in range(0,N//2):
            expo=np.exp((-2*np.pi*1j*k)/N)
            ans[k]=left[k]+expo*right[k]
            ans[k+N//2]=left[k]-expo*right[k]
    return ans

def cooley_fft(serie):
    N=len(serie)
    if (N & (N-1)):
        print(N)
        raise Exception('Series length must be a power of 2')
    return __fft_rec(serie,N)


