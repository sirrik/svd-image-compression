import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def svd():
    f = misc.imread('data.jpg')
    channel1 = f[:,:,2]
    channel2 = f[:,:,1]
    channel3 = f[:,:,0]
    r = np.ndarray((1406,2500))
    g = np.ndarray((1406,2500))
    b = np.ndarray((1406,2500))

    for i in range(1406):
        for j in range(2500):
            r[i][j] = channel1[i][j]
            g[i][j] = channel2[i][j]
            b[i][j] = channel3[i][j]

    r_rt = np.matmul(r,np.transpose(r))
    g_gt = np.matmul(g,np.transpose(g))
    b_bt = np.matmul(b,np.transpose(b))

    eigen_r = np.linalg.eig(r_rt)
    U_r = eigen_r[1]
    U_r_sigma = np.sqrt(np.abs(eigen_r[0]))
    U_r_sigma = np.sort(U_r_sigma)
    U_r_sigma = np.diag(U_r_sigma[::-1])
    eigen_g = np.linalg.eig(g_gt)
    U_g = eigen_g[1]
    U_g_sigma = np.sqrt(np.abs(eigen_g[0]))
    U_g_sigma = np.sort(U_g_sigma)
    U_g_sigma = np.diag(U_g_sigma[::-1])
    eigen_b = np.linalg.eig(b_bt)
    U_b = eigen_b[1]
    U_b_sigma = np.sqrt(np.abs(eigen_b[0]))
    U_b_sigma = np.sort(U_b_sigma)
    U_b_sigma = np.diag(U_b_sigma[::-1])

    rt_r = np.matmul(np.transpose(r),r)
    gt_g = np.matmul(np.transpose(g),g)
    bt_b = np.matmul(np.transpose(b),b)

    eigen_rt = np.linalg.eig(rt_r)
    V_r = eigen_rt[1]
    eigen_gt = np.linalg.eig(gt_g)
    V_g = eigen_gt[1]
    eigen_bt = np.linalg.eig(bt_b)
    V_b = eigen_bt[1]
    U_r_sigma = np.pad(U_r_sigma,((0,0),(0,1094)),mode='constant')
    U_g_sigma = np.pad(U_g_sigma,((0,0),(0,1094)),mode='constant')
    U_b_sigma = np.pad(U_b_sigma,((0,0),(0,1094)),mode='constant')

    last_img = np.ndarray((1406,2500,3),dtype=np.int64)
    last_r = np.matmul(np.matmul(U_r,U_r_sigma),np.transpose(V_r))
    last_g = np.matmul(np.matmul(U_g,U_g_sigma),np.transpose(V_g))
    last_b = np.matmul(np.matmul(U_b,U_b_sigma),np.transpose(V_b))
    # last_b = last_b.astype(int)
    # last_g = last_g.astype(int)
    # last_r = last_r.astype(int)
    last_img[:,:,0] = last_b.real.astype(int)
    last_img[:,:,1] = last_g.real.astype(int)
    last_img[:,:,2] = last_r.real.astype(int)
    print("S(Sigma):",U_r_sigma)
    print(last_img)

    # print(U_r_sigma)
    # print(U_b_sigma)
    # print(U_g_sigma)
    # plt.imshow(last_img)
    # plt.show()

    # print(np.matmul(np.matmul(U_r,U_r_sigma),np.transpose(V_r)))
    # print(np.pad(U_r_sigma,((0,0),(0,1094)),mode='constant'))
    # print(len(np.pad(U_r_sigma,((0,0),(0,1094)),mode='constant')))
    # print(len(np.pad(U_r_sigma,((0,0),(0,1094)),mode='constant')[0]))

def svd_with_rank():
    f = misc.imread('data.jpg')

    channel1 = f[:, :, 2]
    channel2 = f[:, :, 1]
    channel3 = f[:, :, 0]
    r = np.ndarray((1406, 2500))
    g = np.ndarray((1406, 2500))
    b = np.ndarray((1406, 2500))

    for i in range(1406):
        for j in range(2500):
            r[i][j] = channel1[i][j]
            g[i][j] = channel2[i][j]
            b[i][j] = channel3[i][j]

    r_rt = np.matmul(r, np.transpose(r))
    g_gt = np.matmul(g, np.transpose(g))
    b_bt = np.matmul(b, np.transpose(b))
    rank = 50
    U_r = np.ndarray((1406, rank))
    U_b = np.ndarray((1406, rank))
    U_g = np.ndarray((1406, rank))
    U_r_sigma = []
    U_b_sigma = []
    U_g_sigma = []
    eigen_r = np.linalg.eig(r_rt)
    eigen_g = np.linalg.eig(g_gt)
    eigen_b = np.linalg.eig(b_bt)
    for j in range(1406):
        for i in range(rank):
            U_r[j][i] = eigen_r[1][j][i]
            U_g[j][i] = eigen_g[1][j][i]
            U_b[j][i] = eigen_b[1][j][i]
    for i in range(rank):
        U_r_sigma.append(eigen_r[0][i])
        U_g_sigma.append(eigen_g[0][i])
        U_b_sigma.append(eigen_b[0][i])
    U_r_sigma = np.sort(np.sqrt(np.abs(U_r_sigma)))
    U_r_sigma = np.diag(U_r_sigma[::-1])
    U_g_sigma = np.sort(np.sqrt(np.abs(U_g_sigma)))
    U_g_sigma = np.diag(U_g_sigma[::-1])
    # print(U_g_sigma)
    U_b_sigma = np.sort(np.sqrt(np.abs(U_b_sigma)))
    U_b_sigma = np.diag(U_b_sigma[::-1])
    rt_r = np.matmul(np.transpose(r), r)
    gt_g = np.matmul(np.transpose(g), g)
    bt_b = np.matmul(np.transpose(b), b)
    V_r = []
    V_b = []
    V_g = []

    eigen_bt = np.linalg.eig(bt_b)
    eigen_gt = np.linalg.eig(gt_g)
    eigen_rt = np.linalg.eig(rt_r)
    for i in range(rank):
        V_r.append(np.transpose(eigen_rt[1])[i])
        V_g.append(np.transpose(eigen_gt[1])[i])
        V_b.append(np.transpose(eigen_bt[1])[i])
    last_img = np.ndarray((1406, 2500, 3), dtype=np.int64)
    # print(len(U_r))
    # print(len(U_r[0]))
    # print(len(U_r_sigma))
    # print(len(U_r_sigma[0]))
    # print(len(V_r))
    # print(len(V_r[0]))

    last_r = np.matmul(np.matmul(U_r, U_r_sigma), V_r)
    last_g = np.matmul(np.matmul(U_g, U_g_sigma), V_g)
    last_b = np.matmul(np.matmul(U_b, U_b_sigma), V_b)
    last_img[:, :, 0] = last_b.real.astype(int)
    last_img[:, :, 1] = last_g.real.astype(int)
    last_img[:, :, 2] = last_r.real.astype(int)
    plt.imshow(last_img)
    plt.show()