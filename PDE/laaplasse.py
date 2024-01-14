"""ラプラス方程式"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numba import njit
import time




@njit(cache=True)
def main(N : int, M : int):
    """自分で書いた反復法"""
    I_MAX = 10000
    epsilon = 1e-4
    
    U = np.zeros((N, M))
    U[0:2, :] = 0
    U[N-2:N, :] = 0
    U[:, 0:2] = 0
    U[:, M-2:M] = 1
    
    #print(U)
    
    for k in range(I_MAX):
        #print("k = ", k)
        old_U = U.copy()
        for i in range(2, N-2):
            for j in range(2, M-2):
                U[i,j] = 1/4 * (U[i+2,j] + U[i-2,j] + U[i,j+2] + U[i,j-2])
        
        if np.absolute(old_U - U).sum() / (np.absolute(U).sum()+1e-8) < epsilon:
            print("OK")
            print(U)
            break
    
    return U



def main2(N : int, M : int):
    """疎行列使用"""
    
    # 境界条件を作成
    
    _B = np.zeros((N, M))
    _B[0:2, :] = 0
    _B[N-2:N, :] = 0
    _B[:, 0:2] = 0
    _B[:, M-2:M] = 1
    
    
    A_d = []
    A_r = []
    A_c = []
    
    b_d = []
    b_r = []
    b_c = []
    
    _b = 0  # 境界条件値
    
    for j in range(2, N-2):
        for i in range(2, M-2):
            _b = 0
            
            if 4 <= i <= N-4 and 4 <= j <= M-4: # 境界条件関係無し
                if i + N*j - 2*N >= 0:
                    A_r.append(i + N*j)
                    A_c.append(i + N*j - 2*N)
                    A_d.append(1)
                
                if i + N*j - 2 >=0:
                    A_r.append(i + N*j)
                    A_c.append(i + N*j - 2)
                    A_d.append(1)
                
                A_r.append(i + N*j)
                A_c.append(i + N*j)
                A_d.append(-4)

                if i + N*j + 2 < N*M:
                    A_r.append(i + N*j)
                    A_c.append(i + N*j + 2)
                    A_d.append(1)
                
                if i + N*j + 2*N < N*M:
                    A_r.append(i + N*j)
                    A_c.append(i + N*j + 2*N)
                    A_d.append(1)
            
            else:  # 境界条件混入
                if i-2 <= 2:
                    _b += _B[i-2, j]
                else:
                    A_r.append(i + N*j)
                    A_c.append(i + N*j - 2*N)
                    A_d.append(1)
                
                if i+2 >= N-2:
                    _b += _B[i+2, j]
                else:
                    A_r.append(i + N*j)
                    A_c.append(i + N*j + 2*N)
                    A_d.append(1)
                
                if j-2 <= 2:
                    _b += _B[i, j-2]
                else:
                    A_r.append(i + N*j)
                    A_c.append(i + N*j - 2)
                    A_d.append(1)
                
                if j+2 >= M-2:
                    _b += _B[i, j+2]
                else:
                    A_r.append(i + N*j)
                    A_c.append(i + N*j + 2)
                    A_d.append(1)
                
                print(_b)
                if _b != 0:
                    b_r.append(i + N*j)
                    b_c.append(0)
                    b_d.append(-_b)

    print(A_r)
    print(A_c)
    
    dim = N * M
    #dim = (N-2) * (M-2)
    A = sparse.csr_matrix(
        (A_d, (A_r, A_c)),
        shape=((dim, dim))
    )  # 係数行列
    
    b = sparse.csr_matrix(
        (b_d, (b_r, b_c)),
        shape=((dim, 1))
    )  # 係数行列
    
    plt.spy(A)
    # #plt.spy(b)
    plt.show()
    
    x = spsolve(A, b)
    print(x)
    return x





if __name__ == "__main__":
    N = 10
    M = 10
    
    # t0 = time.time()
    # U = main(N, M)
    # print("time = ", time.time() - t0)

    # X, Y = np.mgrid[0:N, 0:M]
    
    # fig = plt.figure(figsize=(9, 9), facecolor="w")
    # ax = fig.add_subplot(111, projection="3d")

    # surf = ax.plot_surface(X, Y, U, cmap="plasma")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # plt.show()
    
    
    main2(N, M)