import numpy as np
import open3d as o3d

def PCA(data):
    # SVD
    U,sigma,VT = np.linalg.svd(data)
    eigenvalues = sigma
    eigenvectors = np.transpose(VT)
    return eigenvalues, eigenvectors

def estimate_plane_and_count_outlers(data, all_data, threshold = 0.1):
    # threshold = 0.4
    # filter outliers
    w_nn, v_nn = PCA(data)
    point_cloud_main_normal = np.transpose(v_nn[:, 2])
    projections_on_normal = np.dot(data, point_cloud_main_normal)
    mean_distance = np.mean(projections_on_normal)

    projections_on_normal = np.dot(all_data, point_cloud_main_normal)
    floor_flag = np.logical_and([projections_on_normal <= mean_distance + threshold], [projections_on_normal >= mean_distance - threshold])
    #print(projections_on_normal.shape, floor_flag.shape)
    inliers = projections_on_normal[floor_flag[0]]
    distance = np.mean(np.abs(inliers))
    return point_cloud_main_normal, mean_distance, distance, np.sum(floor_flag)

def ground_segmentation(data, final_threshold = 0.2, num_iteration = 100):
    # normalization of the data
    offset = np.mean(data ,0)
    data = data - offset

    #num_iteration = 100
    num_sample_points = 5

    closest_distance = 999
    max_inliers = 0
    best_normal = 0
    best_mean = 0

    for i in range(num_iteration):
        init_ids = np.random.choice(data.shape[0], num_sample_points, replace=False)
        samples = data[init_ids]

        point_cloud_main_normal, mean_distance, score_distance , n_inliers = estimate_plane_and_count_outlers(samples, data, final_threshold)

        if(np.abs(point_cloud_main_normal[1]) > 0.2):
            continue

        n_inliers = n_inliers # * np.sqrt(np.abs(point_cloud_main_normal[2]))
        if(n_inliers > max_inliers):
            closest_distance = score_distance
            best_normal = point_cloud_main_normal
            best_mean = mean_distance
            max_inliers = n_inliers
            #print("iteration , ", i , " closest_distance:", closest_distance, n_inliers)

    return max_inliers, best_normal, best_mean, offset

def get_cloud(image, depth, mask, camera_mtx, b_flip = False):
    # estimate the floor function
    camera_mtx_inv = np.linalg.inv(camera_mtx)
    cloud = []
    colors = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if(mask[i,j] and depth[i,j] > 0.1 and depth[i,j] < 8.0):
                pt = np.array([i,j,1])
                pt = np.dot(camera_mtx_inv, np.transpose(pt)) * depth[i,j]
                pt = pt.reshape(3)
                if(b_flip):
                    pt[1] = -pt[1]
                cloud.append(pt)
                color = image[i,j,:]/255
                colors.append([color[2], color[1], color[0]])
    return cloud, colors

def get_floor_cloud(depth, floor_mask, camera_mtx):
    # estimate the floor function
    camera_mtx_inv = np.linalg.inv(camera_mtx)
    cloud = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if(not floor_mask[i,j] and depth[i,j] > 0.1 and depth[i,j] < 5.0):
                pt = np.array([i,j,1])
                pt = np.dot(camera_mtx_inv, np.transpose(pt)) * depth[i,j]
                cloud.append(pt.reshape(3))

    # fill the mask area with the floor
    return cloud

def fill_the_floor_depth(depth, mask, normal, mean, offset, camera_mtx_inv):
    tmp = mean + np.dot(offset, normal)
    result = depth.copy()
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if(not mask[i,j] and depth[i,j] == 0):
                pt = np.array([i,j,1])
                pt = np.dot(camera_mtx_inv, np.transpose(pt))
                dep = tmp / (np.dot(pt, normal))
                result[i,j] = np.abs(dep)
    return result

def fill_the_floor_depth_o3d(depth, depth_raw, mask, normal, mean, offset, camera_mtx_inv):
    tmp = mean + np.dot(offset, normal)
    result = depth.copy()
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if(not mask[i,j] and depth[i,j] == 0):
                pt = np.array([i,j,1])
                pt = np.dot(camera_mtx_inv, np.transpose(pt))
                dep = tmp / (np.dot(pt, normal))
                #result[i,j] = np.abs(dep)
                if(depth_raw[i,j] == 0):
                    result[i,j] = np.abs(dep)
                elif(np.abs(depth_raw[i,j]-dep) > 0.02):
                    result[i,j] = np.abs(dep)
    return result

def floor_repairment(mvs_depth, mask, camera_mtx, iteration = 100, b_fill = False):
    # we need to decide :
    #   1. only delete point under the ground
    #   2. fill the floor mask with the plane model
    n_floor = mask.shape[0] * mask.shape[1] - np.sum(mask)
    if(n_floor < 14000):
        # too few points in the floor mask, alors skip this frame
        return mvs_depth

    # extact the depth data as point cloud
    cloud = get_floor_cloud(mvs_depth, mask, camera_mtx)
    if(len(cloud) < 1000):
        return mvs_depth

    # RANSAC extract the floor
    threshold  = 0.01
    max_score, best_normal, best_mean, offset = ground_segmentation(cloud, threshold, int(iteration))
    if(max_score < 100):
        return mvs_depth

    if(best_normal[0] < 0):
        best_normal = - best_normal

    # delete all the points below the plane
    camera_mtx_inv = np.linalg.inv(camera_mtx)
    result = mvs_depth.copy()
    # TODO: we should use the matrix operations to accelerate this process
    for i in range(mvs_depth.shape[0]):
        for j in range(mvs_depth.shape[1]):
            if(mask[i,j] or mvs_depth[i,j] < 0.1):
                continue
            pt = np.array([i,j,1]) * mvs_depth[i,j]
            pt = np.dot(camera_mtx_inv, np.transpose(pt))
            projections_on_normal = np.dot(pt-offset, best_normal)
            if(projections_on_normal - best_mean > threshold):
                result[i,j] = 0

    if(b_fill):
        result = fill_the_floor_depth(result, mask, best_normal, best_mean, offset, camera_mtx_inv)
    return result

def floor_repairment_o3d(mvs_depth, mask, camera_mtx, b_show):
    n_floor = mask.shape[0] * mask.shape[1] - np.sum(mask)
    if(n_floor < 5000):
        # too few points in the floor mask, alors skip this frame
        return mvs_depth

    cloud_floor = get_floor_cloud(mvs_depth, mask, camera_mtx)
    if(len(cloud_floor) < 1000):
        return mvs_depth

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(cloud_floor)

    #point_cloud_o3d = point_cloud_o3d.voxel_down_sample(voxel_size=0.003)
    #point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    #cl, ind = point_cloud_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #point_cloud_o3d = point_cloud_o3d.select_by_index(ind)

    plane_model, inliers = point_cloud_o3d.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=300)
    [a, b, c, d] = plane_model
    #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    if(b_show):
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = point_cloud_o3d.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = point_cloud_o3d.select_by_index(inliers, invert=True)
        return inlier_cloud, outlier_cloud

    result = mvs_depth.copy()
    result[np.logical_not(mask)] = 0.0
    camera_mtx_inv = np.linalg.inv(camera_mtx)
    result = fill_the_floor_depth_o3d(result, mvs_depth, mask, np.array([a,b,c]), d, np.zeros(3), camera_mtx_inv)
    return result

def delete_underground_points(mvs_depth, camera_mtx):
    cloud_floor = get_floor_cloud(mvs_depth, np.zeros(mvs_depth.shape), camera_mtx)

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(cloud_floor)

    point_cloud_o3d = point_cloud_o3d.voxel_down_sample(voxel_size=0.003)
    point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    cl, ind = point_cloud_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    point_cloud_o3d = point_cloud_o3d.select_by_index(ind)

    plane_model, inliers = point_cloud_o3d.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=300)
    [a, b, c, mean] = plane_model
    #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    result = mvs_depth.copy()
    camera_mtx_inv = np.linalg.inv(camera_mtx)
    #result = fill_the_floor_depth_o3d(result, mvs_depth, mask, np.array([a,b,c]), d, np.zeros(3), camera_mtx_inv)
    normal = np.array([a,b,c])
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if(result[i,j] != 0):
                pt = np.array([i,j,1])
                pt = np.dot(camera_mtx_inv, np.transpose(pt))
                dep = - mean / (np.dot(pt, normal))
                #print(np.abs(result[i,j]), dep)
                if(result[i,j]-dep > -0.08):
                    result[i,j] = 0.0
    return result

    return result

############## some test ###################

# camera_parameters = [640,480,500,500,320,240]
# camera_mtx = np.array([[camera_parameters[2],0,camera_parameters[4]],
#                                 [0,camera_parameters[3],camera_parameters[5]],
#                                 [0,0,1]])
#
# cloud, colors = get_cloud(rgb, mvs_depth, mask, camera_mtx)
#
# point_cloud_o3d_i = o3d.geometry.PointCloud()
# point_cloud_o3d_i.points = o3d.utility.Vector3dVector(cloud)
# point_cloud_o3d_i.colors = o3d.utility.Vector3dVector(colors)


# cloud = get_floor_cloud(mvs_depth, mask, camera_mtx)
#
# threshold  = 0.04
# closest_distance, best_normal, best_mean, offset = ground_segmentation(cloud, threshold, 200)
#
# cloud = np.array(cloud)
# projections_on_normal = np.dot(cloud-offset, best_normal)
# floor_flag = np.logical_and([projections_on_normal <= best_mean + threshold], [projections_on_normal >= best_mean - threshold])
# floor_cloud = cloud[floor_flag[0]]
#
# point_cloud_o3d = o3d.geometry.PointCloud()
# point_cloud_o3d.points = o3d.utility.Vector3dVector(floor_cloud)
# colors = [(0,0,1) for i in range(floor_cloud.shape[0])]
# point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
#
# axis_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([axis_o3d, point_cloud_o3d, point_cloud_o3d_i])
