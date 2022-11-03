import numpy as np
import cv2

pt_1 = './pt_2D_1.txt'
pt_2 = './pt_2D_2.txt'


f1 = open(pt_1, 'r')
f2 = open(pt_2, 'r')

pt_1 = []
pt_2 = []

for line in f1.readlines():
    pt_1.append(line.rstrip('\n'))
for line in f2.readlines():
    pt_2.append(line.rstrip('\n'))

pt_1 = pt_1[1:]
pt_2 = pt_2[1:]

img1 = cv2.imread('./image1.jpg')
img2 = cv2.imread('./image2.jpg')


def LIS_eight(a, b):
    
    matrix = np.zeros((len(a),9))
    
    for i in range(len(a)):
        u = float(a[i][0])
        u_ = float(a[i][1])
        v = float(b[i][0])
        v_ = float(b[i][1])
        matrix[i] = np.array([u*u_, u*v_, u, v*u_, v*v_, v, u_, v_, 1])
    #homogeneous linear system
    
    U, S, V_T  = np.linalg.svd(matrix, full_matrices=True)
    x = V_T[:, 8]
    fake_f = np.reshape(x, (3,3))
    # compute real f
    U,S,V_T = np.linalg.svd(fake_f, full_matrices=True)
    f = U @ np.diag([*S[:2], 0]) @ V_T
    
    return f
    
# fundamental matrix     
#LIS_eight(pt_1, pt_2)

# normalized fundamental matrix
def normalized_points(m):
    uv = []
    for i in range(len(m)):
        uv.append([float(m[i][0]), float(m[i][1])])
    uv = np.array(uv)
    mean = np.mean(uv, axis=0)
    center = uv - mean
    scale = np.sqrt(2 * len(m) / np.power(center, 2))
    trans_matrix = np.array(
        [[scale, 0, -mean[0] * scale],
         [0, scale, -mean[1] * scale],
         [0,0,1]
        ],dtype=object
    )
    return uv, trans_matrix
    
uv1, trans_matrix1=normalized_points(pt_1)
uv2, trans_matrix2=normalized_points(pt_2)
points1 = trans_matrix1 * uv1.T
points2 = trans_matrix2 * uv2.T
points = LIS_eight(points1, points2)
    



    
