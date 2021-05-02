#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import matplotlib
from matplotlib import pyplot as plt
from sklearn import linear_model
matplotlib.use('Tkagg')
np.set_printoptions(suppress=True, precision=4)
seed=1000
random.seed(seed)
np.random.seed(seed)


dist = 1.3
npts = 100
nenv_pts = 100
g_noise_gain = 1e-3
g_noise_env = 10

normal_vec = np.array([1, 2, 3])
normal_vec = normal_vec/np.linalg.norm(normal_vec)
print('ground true norm vector:', normal_vec)
print('ground true dist:', dist)

#90deg about Z
Rz = np.array([
[0,-1,0],
[1,0,0],
[0,0,1]]
)

v1 = Rz @normal_vec
v2 = np.cross(v1, normal_vec)
v2 = v2/np.linalg.norm(v2)
v1 = np.cross(v2, normal_vec)
print(v1)
print(v2)


def get_random_env_noise(number_of_points, magnitude):
    p = np.random.uniform(-magnitude, magnitude ,size=(number_of_points,3))
    return p

points_mat = np.zeros((npts, 3))

for i in range(npts):
    point = random.uniform(-1, 1) * v1 + random.uniform(-1, 1) * v2 + dist * normal_vec
    points_mat[i] = point

pts_clean = points_mat.T.copy()
points_noise_mat = g_noise_gain * np.random.uniform(-1, 1, (npts, 3))
points_mat = points_mat + points_noise_mat
points_mat = np.vstack([points_mat, get_random_env_noise(nenv_pts, g_noise_env)])
pts = points_mat.T


#least_squares SVD solving Ax=b
A = pts.T
number_pts, _ = A.shape
B = np.ones((number_pts,1))
U, s, Vh = np.linalg.svd(A, full_matrices=False)
c = np.dot(U.T, B)
w = np.dot(np.diag(1 / s), c)
x = np.dot(Vh.conj().T, w)
estimated_dist_svd = 1/np.linalg.norm(x)
print('[ax=b] estimated dist', estimated_dist_svd)
print('[ax=b] ground true dist:', dist)
estimated_norm_vector_svd = x.T[0]*estimated_dist_svd
print('[ax=b] estimated normal vector:', estimated_norm_vector_svd)
print('[ax=b] ground true norm vector:', normal_vec)


#RANSAC
xyz = (pts.T)
xy = xyz[:, :2]
z = xyz[:, 2]
ransac = linear_model.RANSACRegressor(residual_threshold=None, max_trials=100, random_state=seed)
ransac.fit(xy, z)
a, b = ransac.estimator_.coef_  # coefficients
d = ransac.estimator_.intercept_  # intercept
inlier_mask = ransac.inlier_mask_
#Z = aX + bY + d
#aX + by - Z = -d
ransac_vec = np.array([a,b,-1])/-d
estimated_dist_ransec = 1/np.linalg.norm(ransac_vec)
print('[ransac] estimated dist', estimated_dist_ransec)
print('[ransac] ground true dist:', dist)
print('[ransac] estimated normal vector:', ransac_vec)
print('[ransac] ground true norm vector:', normal_vec)
#range of grid # match the pts range
min_x, max_x, min_y, max_y = np.min(pts[0])-1, np.max(pts[0])+2, np.min(pts[1])-1, np.max(pts[1])+2
step = 1.5 #grid density lower is dense
X, Y = np.meshgrid(np.arange(min_x, max_x, step=step), np.arange(min_y, max_y, step=step))
Z_2 = np.zeros(X.shape)
Z_ransac = np.zeros(X.shape)
Z_o = np.zeros(X.shape)
original_normal_vec = normal_vec / dist
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        #orignal
        Z_o[r, c] = (1 - (original_normal_vec[0] * X[r, c] + original_normal_vec[1] * Y[r, c])) / original_normal_vec[2]
        #svd
        Z_2[r, c] = (1 - (estimated_norm_vector_svd[0] * X[r, c] + estimated_norm_vector_svd[1] * Y[r, c])) / estimated_norm_vector_svd[2]
        #ransac
        Z_ransac[r,c] = (1 - (ransac_vec[0] * X[r, c] + ransac_vec[1] * Y[r, c])) /ransac_vec[2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
point_scale = 500/npts#point size scale
ax.scatter(pts[0], pts[1], pts[2], color='red', s=point_scale)
ax.scatter(pts_clean[0], pts_clean[1], pts_clean[2], color='green', s=point_scale)
ax.plot_wireframe(X, Y, Z_o, color='green')  # orignal
ax.plot_wireframe(X, Y, Z_2, color='pink')  # ax=b
ax.plot_wireframe(X, Y, Z_ransac, color='black') #ransac

plt.show()
print('npts:', npts)
print('g_noise_gain:', g_noise_gain)
print('ground true dist:', dist)
print('(original plane) ground true norm vector:', normal_vec)
print('(pink plane) estimated normal vector:', estimated_norm_vector_svd.T[0])
print('(black plane) estimated normal vector:', ransac_vec)


vec = normal_vec.T
dists = np.abs(vec @ pts_clean) / np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
avg_dist = np.mean(dists)
print(avg_dist)


vec = estimated_norm_vector_svd
dists = np.abs(vec @ pts_clean) / np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
avg_dist = np.mean(dists)
err = np.abs((avg_dist-dist))*100/dist
err = np.round(err,4)
print (err,'%')

vec = ransac_vec
dists = np.abs(vec @ pts_clean) / np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
avg_dist = np.mean(dists)
err = np.abs((avg_dist-dist))*100/dist
err = np.round(err,4)
print (err,'%')



