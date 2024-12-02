# %%

import cv2
import matplotlib.pyplot as plt
import numpy as np
from functions.matchPics import matchPics
from functions.bundle_adjustment import bundle_adjustment
import os
from functions.plot_utils import viz_3d, viz_3d_matplotlib, draw_epipolar_lines
# %%
def rep_error_fn(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3,4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]

        print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])


def computeF(xl,xr):

    F, mask = cv2.findFundamentalMat(np.array(xl),np.array(xr),cv2.FM_RANSAC)
    return F, mask


def computePose(H, K):

    E = np.matmul(np.matmul(np.transpose(K), H), K)
    retval, R, t, mask = cv2.recoverPose(E, list_kp1, list_kp2, K)
    return retval, R, t, mask

def triangulation(R, t, list_kp1, list_kp2):
    R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    R_t_1 = np.empty((3,4))
    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3,4))

    R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
    R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())
    P2 = np.matmul(K, R_t_1)


    list_kp1 = np.transpose(list_kp1)
    list_kp2 = np.transpose(list_kp2)

    points_3d = cv2.triangulatePoints(P1, P2, list_kp1, list_kp2)
    points_3d /= points_3d[3]
    return points_3d, P2


# %%
import os
iter = 0
prev_img = None
prev_kp = None
prev_desc = None
K = np.loadtxt('data/templeRing/camera.txt')
R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
R_t_1 = np.empty((3,4))
P1 = np.matmul(K, R_t_0)
P2 = np.empty((3,4))
pts_4d = []
X = np.array([])
Y = np.array([])
Z = np.array([])

for filename in os.listdir('data/templeRing')[0:3]:
    print(filename)

    file = os.path.join('data/templeRing', filename)
    img = cv2.imread(file)

    resized_img = img
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(resized_img,None)

    if iter == 0:
        prev_img = resized_img
        prev_kp = kp
        prev_desc = desc
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(prev_desc,desc,k=2)
        good = []
        list_kp1 = []
        list_kp2 = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append(m)
                list_kp1.append(prev_kp[m.queryIdx].pt)
                list_kp2.append(kp[m.trainIdx].pt)

        list_kp1 = np.array(list_kp1)
        list_kp2 = np.array(list_kp2)

        F,mask = computeF(list_kp1, list_kp2)
        list_kp1 = list_kp1[mask.ravel()==1]
        list_kp2 = list_kp2[mask.ravel()==1]

        K = np.loadtxt('data/templeRing/camera.txt')
        retval, R, t, mask = computePose(F, K)

        points_3d, P2 = triangulation(R, t, list_kp1, list_kp2)

        list_kp1 = np.transpose(list_kp1)
        list_kp2 = np.transpose(list_kp2)

        X = np.array([])
        Y = np.array([])
        Z = np.array([])

        opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
        num_points = len(list_kp2[0])
        rep_error_fn(opt_variables, list_kp2, num_points)

        X = np.concatenate((X, points_3d[0]))
        Y = np.concatenate((Y, points_3d[1]))
        Z = np.concatenate((Z, points_3d[2]))

        R_t_0 = np.copy(R_t_1)
        P1 = np.copy(P2)
        prev_img = resized_img
        prev_kp = kp
        prev_desc = desc

    iter = iter + 1
pts_4d.append(X)
pts_4d.append(Y)
pts_4d.append(Z)

viz_3d(np.array(pts_4d))
