import numpy as np
import pye57

from pyquaternion import Quaternion
from panorama_to_pinhole import *

################ camera tools ####################

def camera_fov(fx, width):
    side = width * 0.5 / fx
    return np.arctan2(side, 1) * 360 / np.pi


def color_jet(depth_image):
    max_d = np.max(depth_image)
    #print(max_d)
    depth_jet = np.zeros([depth_image.shape[0],depth_image.shape[1],3]).astype(np.uint8)
    
    norm_depth = depth_image / max_d
    norm_depth = norm_depth.reshape([depth_image.shape[0],depth_image.shape[1]])
    depth_jet[:,:,0] = norm_depth * 255
    depth_jet[:,:,2] = 255 - depth_jet[:,:,0]
    return depth_jet

def get_depth(cloud_pt, camera_mtx, image_w_h, P):
    image = np.zeros(image_w_h)
    cam_R = P[0:3,0:3]
    cam_t = P[0:3, 3].reshape([3,1])
    #print(cam_R,cam_t)
    for i in range(len(cloud_pt)):
        pt = cloud_pt[i].reshape([3,1])
        pt_camera = np.dot( camera_mtx, np.dot(cam_R, pt) + cam_t )
        if(pt_camera[2] < 0.1):
            continue
            
        img_u = int(pt_camera[1]/pt_camera[2])
        img_v = int(pt_camera[0]/pt_camera[2])
        
        if img_u < 0 or img_u > image_w_h[0]-1:
            continue
        if img_v < 0 or img_v > image_w_h[1]-1:
            continue 
        
        image[img_u,img_v] = pt_camera[2]
    return image

def get_depth_pinhole(cloud_pt, camera_mtx, image_w_h, P):
    image = np.zeros(image_w_h)
    cam_R = P[0:3,0:3]
    cam_t = P[0:3, 3].reshape([3,1])
    
    pt = cloud_pt.transpose()
    pt_camera = np.dot(camera_mtx, np.dot(cam_R, pt) + cam_t )
    #print(pt_camera.shape)
    pt_camera = pt_camera[:,pt_camera[2,:] > 0.1]
    img_u = (pt_camera[1,:]/pt_camera[2,:]).astype(int)
    img_v = (pt_camera[0,:]/pt_camera[2,:]).astype(int)
    
    flag = (img_u > 0) & (img_u < image_w_h[0]-1) & (img_v > 0) & (img_v < image_w_h[1]-1)
    img_u = img_u[flag]
    img_v = img_v[flag]
    depth = pt_camera[2,:][flag]
    image[img_u, img_v] = depth
    
    return image

############### point cloud tools ######################


def Transform_cloud(cloud, R, t):
    new_cloud = np.dot(R, np.transpose(cloud)) + np.reshape(t, [3,1])
    return new_cloud.transpose()


def project_points_to_panorama(cloud, colors, height = 1280):
    width = height*2
    image = np.zeros([height, width,3]).astype(np.uint8)
    depth = np.zeros([height, width])
    
    depth_all = np.linalg.norm(cloud, axis=1)[:, None]
    norm_cloud = cloud / depth_all
    sph_pitch = np.arcsin(norm_cloud[:, 1]) 
    sph_yaw = np.arctan2(norm_cloud[:, 0], norm_cloud[:, 2])
    
    pt_u = (width * (sph_yaw /np.pi)/2 + width/2).astype(int)
    pt_v = (height * (sph_pitch/np.pi) + height/2).astype(int)
    
    for i in range(len(cloud)):        
        if(pt_u[i] < 0 or pt_u[i] > width-1):
            continue
        if(pt_v[i] < 0 or pt_v[i] > height-1):
            continue
        
        #cv2.circle(image,(pt_u,pt_v),3,color,-1)
        image[pt_v[i], pt_u[i], :] = colors[i]
        depth[pt_v[i], pt_u[i]] = depth_all[i]
        
    return image, depth

def panorama_to_cloud(image, depth, interval=1):
    image_uni = cv2.resize(image, (depth.shape[1], depth.shape[0]))
    #print(image_uni.shape, depth.shape)
    width = depth.shape[1]
    height = depth.shape[0]
    points = []
    colors = []

    for i in range(0,depth.shape[0],interval):
        pitch = (i - height/2) * np.pi / height 
        for j in range(0,depth.shape[1],interval):
            if(depth[i,j] < 0.1):
                continue
            yaw = (j - width/2) * 2 * np.pi / width
            sphere = np.array([np.cos(pitch)*np.sin(yaw) , np.sin(pitch), np.cos(pitch)*np.cos(yaw)])
            point_3d = sphere * depth[i,j]
            points.append(point_3d)
            colors.append(image_uni[i,j,:])
            
    return points, colors

################## e57 tools ####################

# real e57 file
def read_e57_with_pose(pano_scan, bTransform = True):
    e57 = pye57.E57(pano_scan) # read scan at index 0
    data = e57.read_scan(0,colors=True)
    
    n_pts = data["cartesianX"].shape[0]
    colors = np.zeros((n_pts,3))
    colors[:,0] = data["colorRed"]
    colors[:,1] = data["colorGreen"]
    colors[:,2] = data["colorBlue"]
    
    points = np.zeros((n_pts,3))
    points[:,0] = data["cartesianX"]
    points[:,1] = data["cartesianY"]
    points[:,2] = data["cartesianZ"]
    
    imf = e57.image_file
    root = imf.root()
    data3d = root["data3D"]
    scan_0 = data3d[0]
    translation_x = scan_0["pose"]["translation"]["x"].value()
    translation_y = scan_0["pose"]["translation"]["y"].value()
    translation_z = scan_0["pose"]["translation"]["z"].value()
    
    rotation_x = scan_0["pose"]["rotation"]["x"].value()
    rotation_y = scan_0["pose"]["rotation"]["y"].value()
    rotation_z = scan_0["pose"]["rotation"]["z"].value()
    rotation_w = scan_0["pose"]["rotation"]["w"].value()
    
    rotation = [rotation_w, rotation_x, rotation_y, rotation_z]
    translation = [translation_x, translation_y, translation_z]
    q = Quaternion(rotation)
    R_scan = np.transpose(q.rotation_matrix)
    t_scan = - np.dot(R_scan ,np.transpose(np.asarray(translation)))
    
    if(bTransform) :
        points_scan_transformed = Transform_cloud(points, R_scan, t_scan)
    
        ## rotate 90 degree around x axis
        q_x90 = Quaternion(axis = [1,0,0], angle=np.pi/2)
        points_scan_transformed_f = Transform_cloud(points_scan_transformed, q_x90.rotation_matrix, np.zeros(3))

        ## rotate 90 degree around y axis
        q_y90 = Quaternion(axis = [0,1,0], angle=np.pi/2)
        points_scan_transformed_f = Transform_cloud(points_scan_transformed_f, q_y90.rotation_matrix, np.zeros(3))
    else:
        points_scan_transformed_f = points

    return points_scan_transformed_f, colors, R_scan, t_scan


def test_e57(pano_scan):
    e57 = pye57.E57(pano_scan)# read scan at index 0
    data = e57.read_scan(0)

    # 'data' is a dictionary with the point types as keys
    assert isinstance(data["cartesianX"],np.ndarray)
    assert isinstance(data["cartesianY"],np.ndarray)
    assert isinstance(data["cartesianZ"],np.ndarray)

    # other attributes can be read using:
    data = e57.read_scan(0,intensity=True,colors=True,row_column=True)
    assert isinstance(data["cartesianX"],np.ndarray)
    assert isinstance(data["cartesianY"],np.ndarray)
    assert isinstance(data["cartesianZ"],np.ndarray)
    assert isinstance(data["intensity"],np.ndarray)
    assert isinstance(data["colorRed"],np.ndarray)
    assert isinstance(data["colorGreen"],np.ndarray)
    assert isinstance(data["colorBlue"],np.ndarray)
    assert isinstance(data["rowIndex"],np.ndarray)
    assert isinstance(data["columnIndex"],np.ndarray)

    # the 'read_scan' method filters points using the 'cartesianInvalidState' field
    # if you want to get everything as raw, untransformed data, use:
    data_raw = e57.read_scan_raw(0)

    # writing is also possible, but only using raw data for now
    e57_write=pye57.E57("e57_file_write.e57",mode='w')
    e57_write.write_scan_raw(data_raw)

    # you can specify a header to copy information from
    e57_write.write_scan_raw(data_raw,scan_header=e57.get_header(0))

    # the ScanHeader object wraps most of the scan information:
    header=e57.get_header(0)
    print(header.point_count)
    print(header.rotation_matrix)
    print(header.translation)

    # all the header information can be printed using:
    for line in header.pretty_print():
        print(line)
    
    # the scan position can be accessed with:
    position_scan_0 = e57.scan_position(0)

    # the binding is very close to the E57Foundation API
    # you can modify the nodes easily from python
    imf = e57.image_file
    root = imf.root()
    data3d = root["data3D"]
    scan_0 = data3d[0]
    translation_x=scan_0["pose"]["translation"]["x"]
