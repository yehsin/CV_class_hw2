import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import math

def homography(src_pts, tar_pts):
    trans_matrix = []
    dest = np.array([tar_pts[0][0],tar_pts[0][1], tar_pts[1][0], tar_pts[1][1], tar_pts[2][0], tar_pts[2][1], tar_pts[3][0], tar_pts[3][1]])

    for i in range(src_pts.shape[0]):
        x, y = src_pts[i]
        x_hat, y_hat = tar_pts[i]
        trans_matrix.append([x,y,1.0,0.0,0.0,0.0,-x*x_hat, -y*x_hat])
        trans_matrix.append([0.0,0.0,0.0,x,y,1.0,-x*y_hat, -y*y_hat])
    trans_matrix = np.array(trans_matrix)
    h = np.linalg.lstsq(trans_matrix, dest)[0]
    h = np.concatenate((h, [1]), axis=-1)
    h = np.reshape(h, (3,3))

    return h


img = cv2.imread('./Delta-Building.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

source_points = np.array([[425,330],[400,800],[900,1000],[880,0]])
target_points = np.array([[0,0],[0,512],[512,512],[512,0]])
h = homography(source_points, target_points)

# project
new_img = np.zeros((512, 512, 3))
# backward warping
# from destination positions find source positions, and do bilinear interpolation to copy pixels to destination image
for heigh in range(512):
    for width in range(512):
        project_pt = np.array([heigh, width, 1])
        ori_value = np.linalg.inv(h) @ project_pt.T
        ori_value = ori_value / ori_value[2]
        u,v = ori_value[0], ori_value[1]

        # bilinear interpolation
        a = math.ceil(u)
        b = math.floor(u)
        c = math.ceil(v)
        d = math.floor(v)

        new_img[width,heigh] = (img[d,b] * (a-u) + img[d,a] * (u-b)) * (c-v) + (img[c,b] * (a-u) + img[c,a] * (u-b)) * (v-d)

figure = plt.figure(figsize=(12, 6))
subplot1 = figure.add_subplot(1, 2, 1)
subplot1.title.set_text("Source Image")
subplot1.imshow(img)

subplot2 = figure.add_subplot(1, 2, 2)
subplot2.title.set_text("Destination Image")
subplot2.imshow(new_img.astype('uint8'))
plt.show()

"""
Experiment forward mapping
"""
# for heigh in range(img.shape[0]):
#     for width in range(img.shape[1]):
#         project_pt = np.array([heigh, width, 1])
#         ori_value = h @ project_pt.T
#         ori_value = ori_value / ori_value[2]
#         u,v = ori_value[0], ori_value[1]
#         if (u >= 0 and u < 512 and v >= 0 and v < 512):
#             a = math.floor(u)
#             c = math.floor(v)
#             new_img[c,a] = img[width, heigh]
        

# figure = plt.figure(figsize=(12, 6))
# subplot1 = figure.add_subplot(1, 2, 1)
# subplot1.title.set_text("Source Image")
# subplot1.imshow(img)

# subplot2 = figure.add_subplot(1, 2, 2)
# subplot2.title.set_text("Destination Image")
# subplot2.imshow(new_img.astype('uint8'))
# plt.show()



