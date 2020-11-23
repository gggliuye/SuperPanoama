import numpy as np
import imageio
import cv2
import glob
import os

import open3d as o3d
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

import skimage
from skimage.filters import sobel
from skimage.feature import canny
from skimage.segmentation import watershed
from scipy import ndimage as ndi

def rank_images(images):
    ids = []
    for image in images:
        id_str = image.split('\\')[-1][:-4]
        ids.append(int(id_str))

    rank = np.argsort(np.asarray(ids))
    return [images[i] for i in rank]

def q_t_to_matrix(q_vec, t_vec):
    quaternion = Quaternion(q_vec)
    pose = np.eye(4)
    pose[0:3,0:3] = quaternion.rotation_matrix
    pose[0:3, 3] = t_vec
    #pose = np.linalg.inv(pose)
    return pose

def reshape_depth(depth, size=(480,640)):
    reshaped_depth = np.zeros(size)
    height = min(size[0], depth.shape[0])
    width = min(size[1], depth.shape[1])
    reshaped_depth[0:height, 0:width] = depth[0:height, 0:width]
    return reshaped_depth

def filter_depth(depth, prec = 5):
    # filter by the percentile
    depth_tmp = np.zeros(depth.shape) + depth
    depth_range = np.percentile(depth_tmp[depth_tmp>0.1], [prec,100-prec])

    depth_tmp[depth_tmp < depth_range[0]] = 0
    depth_tmp[depth_tmp > depth_range[1]] = 0

    # filter the closest points
    max_t = np.max(depth_tmp[depth_tmp> 0.01])
    min_t = np.min(depth_tmp[depth_tmp> 0.01])

    threshold = (max_t - min_t)/50 + min_t
    depth[depth < threshold] = 0
    return depth

def show_rgbd_test(rgbd_image):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    plt.subplot(1, 2, 1)
    plt.title('color image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()
    return pcd

def save_points_to_ply(cloud, colors, file_name):
    color = (255*colors).astype(np.int)

    file_out = open(file_name, "w")

    file_out.write('ply\nformat ascii 1.0\ncomment Created by LIUYE\n')
    str_size = 'element vertex ' + str(cloud.shape[0]) + '\n'
    file_out.write(str_size)
    file_out.write('property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')

    for i in range(cloud.shape[0]):
        str_pt = str(cloud[i,0]) + ' ' + str(cloud[i,1]) + ' ' + str(cloud[i,2]) + ' ' + str(color[i,0]) + ' ' + str(color[i,1]) + ' ' + str(color[i,2]) + '\n'
        file_out.write(str_pt)

    file_out.close()

def image_depth_to_pointcloud(image, depth, camera_parameters):
    #if(image.shape[0] != camera_parameters[1]):
    #    print(' ERROR : wrong camera parameters! ')
    #    return None, None

    resize_ratio = depth.shape[0] / image.shape[0]
    resized_image = cv2.resize(image, (depth.shape[1], depth.shape[0]))

    calibration_matrix = np.array([[resize_ratio*camera_parameters[2],0,resize_ratio*camera_parameters[4]],
                                   [0,camera_parameters[3]*resize_ratio,camera_parameters[5]*resize_ratio],
                                   [0,0,1]])
    #print(calibration_matrix)
    inv_calibration = np.linalg.inv(calibration_matrix)
    #print(inv_calibration)
    cloud = []
    colors =[]
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            point_uv = np.array([j, i, 1])
            depth_pt = depth[i,j]
            if(depth_pt < 0.2):
                continue
            point_3d = depth_pt * np.dot(inv_calibration, np.transpose(point_uv))
            color = resized_image[i,j,:]/255
            cloud.append([point_3d[0], point_3d[1], point_3d[2]])
            colors.append([color[2], color[1], color[0]])
    return cloud, colors

def denoise_depth(depth, dilate_iterations=5, b_show=False):
    #elevation_map = sobel(depth)

    edges = canny(depth/20)
    fill_coins = ndi.binary_fill_holes(edges)

    label_objects, nb_labels = ndi.label(fill_coins)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 1
    mask_sizes[0] = 0
    coins_cleaned = mask_sizes[label_objects]

    kernel = np.ones((2,2), dtype=np.uint8)
    coins_cleaned = cv2.dilate(np.array(coins_cleaned).astype(np.float32), kernel,iterations=dilate_iterations)
    coins_cleaned = coins_cleaned == 1

    depth_filtered = depth.copy()
    depth_filtered[coins_cleaned] = 0

    if b_show:
        plt.figure(figsize=(20,8))
        plt.subplot(141); plt.imshow(depth)
        plt.subplot(142); plt.imshow(fill_coins)
        plt.subplot(143); plt.imshow(coins_cleaned)
        plt.subplot(144); plt.imshow(depth_filtered)

    return depth_filtered
