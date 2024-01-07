"""ラプラス方程式"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numba import njit


@njit(cache=True)
def update(U, i, j):
    U[i,j] = 1/4 * (U[i+2,j] + U[i-2,j] + U[i,j+2] + U[i,j-2])
    return





@njit(cache=True)
def main():
    N = 2000
    M = 2000
    I_MAX = 10000
    epsilon = 1e-4
    
    U = np.zeros((N, M))
    U[0:1, :] = 1
    U[N-1:N, :] = 2
    U[:, 0:1] = 3
    U[:, M-1:M] = 4
    
    print(U)
    
    for k in range(I_MAX):
        print("k = ", k)
        old_U = U.copy()
        for i in range(2, N-2):
            for j in range(2, M-2):
                update(U, i, j)
        
        #print(old_U - U)
        if np.absolute(old_U - U).sum() / (np.absolute(U).sum()+1e-8) < epsilon:
            print("OK")
            print(U)
            break
    
    return U






if __name__ == "__main__":
    # U = np.array([[range(25)]])
    # U = U.reshape(5,5)
    # print(U)
    
    # update(U, 2,2)
    # print(U)
    
    U = main()
    X, Y = np.mgrid[0:2000, 0:2000]
    
    fig = plt.figure(figsize=(9, 9), facecolor="w")
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, U, cmap="plasma")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
    
    