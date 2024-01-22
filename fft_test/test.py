import numpy as np
import matplotlib.pyplot as plt



L = 500  # 測定幅
DELTA = 0.1  # サンプリング周期
A = 5

x = np.arange(0, L, DELTA)
z = A * np.sin(x / L  * np.pi)

n = np.random.normal(loc=0, scale=0.2, size=x.shape[0])
z2 = z + n

z3 = np.convolve(z2, 100, mode="full") 



N = x.shape[0]  # データ数
freq = np.linspace(0, 1/DELTA, N)
F = np.fft.fft(z2)
F = np.abs(F)
F = F / (2*N)

plt.plot(freq, F)


# fig = plt.figure()
# ax = fig.add_subplot(2, 2, 1)
# ax.plot(x, z)

# ax2 = fig.add_subplot(2, 2, 2)
# ax2.scatter(x, z2, s=5)

# ax3 = fig.add_subplot(2, 2, 3)
# ax3.plot(x, z3)


plt.show()