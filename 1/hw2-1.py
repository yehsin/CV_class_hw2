import numpy as np
import cv2
import matplotlib.pyplot as plt

pt_1 = './pt_2D_1.txt'
pt_2 = './pt_2D_2.txt'


f1 = open(pt_1, 'r')
f2 = open(pt_2, 'r')

pt_1 = []
pt_2 = []

for line in f1.readlines():
    x = line.rstrip('\n').split(' ')
    pt_1.append(x)
for line in f2.readlines():
    x = line.rstrip('\n').split(' ')
    pt_2.append(x)

pt_1 = pt_1[1:]
pt_2 = pt_2[1:]

pt_1 = np.array(pt_1)
pt_2 = np.array(pt_2)

img1 = cv2.imread('./image1.jpg')
img2 = cv2.imread('./image2.jpg')


def LIS_eight(a, b):
    matrix = np.zeros((a.shape[0],9))

    for i in range(a.shape[0]):
        u = float(a[i][0])
        v = float(a[i][1])
        u_ = float(b[i][0])
        v_ = float(b[i][1])
        matrix[i] = np.array([u*u_, u_*v, u_, v_*u, v*v_, v_, u, v, 1])

    # Decompose ATA
    U, D, V  = np.linalg.svd(matrix,full_matrices=True)
    x = V.T[:, 8]
    F = np.reshape(x, (3,3))
    """
    Code above satisfied F = 1 requirement,
    We still need rank2 requirement
    """
    # compute rank2 f
    FU,FD,FV = np.linalg.svd(F,full_matrices=True)
    F = np.dot(FU, np.dot(np.diag([*FD[:2], 0]), FV))
    
    return F
    
# fundamental matrix
F = LIS_eight(pt_1, pt_2)


# normalized fundamental matrix
def normalized_points(m):
    uv = []
    for i in range(m.shape[0]):
        uv.append([float(m[i][0]), float(m[i][1])])
    uv = np.array(uv)

    # Center
    mean = np.mean(uv, axis=0)
    center = uv - mean

    # Scale
    scale = np.sqrt(2 * len(m) / np.sum(np.power(center, 2)))
    trans_matrix = np.array(
        [[scale, 0, -mean[0] * scale],
         [0, scale, -mean[1] * scale],
         [0,0,1]
        ],dtype=object
    )
    return uv, trans_matrix
    
uv1, trans_matrix1=normalized_points(pt_1)
uv2, trans_matrix2=normalized_points(pt_2)
uv1 = np.insert(uv1,uv1.shape[1],values=1, axis=1)
uv2 = np.insert(uv2,uv2.shape[1],values=1, axis=1)
# q = Tp
points1 = (trans_matrix1 @ (uv1.T)).T
# q = T'p'
points2 = (trans_matrix2 @ (uv2.T)).T
F_norm = LIS_eight(points1, points2)
# T'FT
F_norm = trans_matrix2.T @ (F_norm) @ (trans_matrix1)
#print(points_norm)

# pFp = [points2[i].dot(F_norm.dot(points1[i])) 
#             for i in range(points1.shape[0])]
# print("p'^T F p =", np.abs(pFp).max())

def plot_(pt1, pt2, img1, img2, f):
    plt.subplot(1,2,1)
    # That is epipolar line associated with p.
    ln1 = f.T.dot(pt2.T)
    # Ax + By + C = 0
    A,B,C = ln1
    for i in range(ln1.shape[1]):
        # when y as 0，x = - (C/A)
        # when y = image.shape[0], x = -(Bw + C / A)
        # when x as image.shape[1], y = - (Aw + C / B)
        # when x as 0, y = - (C / B)
        #plt.plot([-C[i]/A[i], img1.shape[1]], [0, -(A[i]*img1.shape[1] + C[i])/B[i]], 'r')
        if ((-C[i]/B[i]) <0):
            plt.plot([-C[i]/A[i],img1.shape[1]],[0, -(C[i] + A[i]*img1.shape[1])/B[i]], 'r')
        elif ((-C[i]/B[i]) > img1.shape[0]):
            plt.plot([-(C[i] + B[i]*img1.shape[0])/A[i],img1.shape[1]],[img1.shape[0], -(C[i] + A[i]*img1.shape[1])/B[i]], 'r')
        else:
            plt.plot([0, img1.shape[1]], [-C[i]/B[i], -(C[i] + A[i]*img1.shape[1])/B[i]], 'r')
        plt.plot([pt1[i][0]], [pt1[i][1]], 'b*')
    plt.imshow(img1, cmap='gray')

    plt.subplot(1,2,2)
    # That is the epipolar line associated with p’.
    ln2 = f.dot(pt1.T)
    # Ax + By + C = 0
    A,B,C = ln2
    for i in range(ln2.shape[1]):
        # when y as 0，x = - (C/A)
        # when y = image.shape[0], x = -(Bw + C / A)
        # when x as image.shape[1], y = - (Aw + C / B)
        # when x as 0, y = - (C / B)
        #plt.plot([-C[i]/A[i], img1.shape[1]], [0, -(A[i]*img1.shape[1] + C[i])/B[i]], 'r')
        if ((-C[i]/B[i]) <0):
            plt.plot([-C[i]/A[i],img2.shape[1]],[0, -(C[i] + A[i]*img2.shape[1])/B[i]], 'r')
        elif ((-C[i]/B[i]) > img2.shape[0]):
            plt.plot([-(C[i] + B[i]*img2.shape[0])/A[i],img2.shape[1]],[img2.shape[0], -(C[i] + A[i]*img2.shape[1])/B[i]], 'r')
        else:
            plt.plot([0, img2.shape[1]], [-C[i]/B[i], -(C[i] + A[i]*img2.shape[1])/B[i]], 'r')
        plt.plot([pt1[i][0]], [pt1[i][1]], 'b*')
    plt.imshow(img2, cmap='gray')
    plt.show()

def plot_norm(pt1, pt2, img1, img2, f):
    plt.subplot(1,2,1)

    # That is epipolar line associated with p.
    ln1 = f.T.dot(pt2.T)
    # Ax + By + C = 0
    A,B,C = ln1
    for i in range(ln1.shape[1]):
        # when x as 0，y = - (C/B)
        # when x as 512(w), y = - (Aw + C / B)
        plt.plot([0, img1.shape[1]], [-C[i]/B[i], -(C[i] + A[i]*img1.shape[1])*1.0/B[i]], 'r')
        plt.plot([pt1[i][0]], [pt1[i][1]], 'b*')
    plt.imshow(img1, cmap='gray')

    plt.subplot(1,2,2)
    # That is the epipolar line associated with p’.
    ln2 = f.dot(pt1.T)
    # Ax + By + C = 0
    A,B,C = ln2
    for i in range(ln2.shape[1]):
        plt.plot([0, img2.shape[1]], [-C[i]*1.0/B[i], -(A[i]*img2.shape[1] + C[i])/B[i]], 'r')
        plt.plot([pt2[i][0]], [pt2[i][1]], 'b*')
    plt.imshow(img2, cmap='gray')
    plt.show()

plot_(uv1, uv2, img1, img2, F)
plot_norm(uv1, uv2, img1, img2, F_norm)


def calaulate_dist(pt1, pt2, f):
    ln1 = f.T.dot(pt2.T)
    pt_num = pt1.shape[0]
    a,b,c = ln1

    dist = 0.0
    for i in range(pt_num):
        dist += np.abs((a[i]*pt1[i][0] + b[i]*pt1[i][1] + c[i])) / np.sqrt(np.power(a[i],2) + np.power(b[i],2))
    acc = dist / pt_num
    return acc

# acc associated with point2
print('Accuracy of the fundamental matrices by point2:', calaulate_dist(uv1, uv2, F))
print('Accuracy of the normalized fundamental matrices by point2:', calaulate_dist(uv1, uv2, F_norm))
# acc associated with point1
print('Accuracy of the fundamental matrices by point1:', calaulate_dist(uv2, uv1, F.T))
print('Accuracy of the normalized fundamental matrices by point1:', calaulate_dist(uv2, uv1, F_norm.T))

    



    
