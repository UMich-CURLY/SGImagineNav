"""Utilities for generating 2D map
"""
import numpy as np
import open3d as o3d
import cv2
from skimage import measure
import skimage.morphology

from utils.fmm_planner import FMMPlanner


def build_full_scene_pcd(depth, image, hfov):
    height, width = depth.shape

    cx = (width - 1.) / 2.
    cy = (height - 1.) / 2.
    fx = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    # fy = (height / 2.) / np.tan(np.deg2rad(self.args.hfov / 2.))

    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # Apply the mask, and unprojection is done only on the valid points
    valid_mask = depth > 0
    masked_depth = depth[valid_mask]
    u = u[valid_mask]
    v = v[valid_mask]

    # Convert to 3D coordinates
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fx
    z = masked_depth

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)
    
    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    color_mask = np.repeat(valid_mask[:, :, np.newaxis], 3, axis=2)
    image_flat = image[color_mask].reshape(-1, 3)  # Flatten the image array for easier indexing
    colors = image_flat / 255.0  # Normalize the colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    camera_object_pcd = pcd.voxel_down_sample(0.05)

    return camera_object_pcd



def detect_frontier(explored_map, obstacle_map, current_pose, threshold_point):
    # ------------------------------------------------------------------
    ##### Get the frontier map and score
    # ------------------------------------------------------------------
    map_size = explored_map.shape[0]
    edge_map = np.zeros((map_size, map_size))
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)) # old version
    dis_obstacle_map = obstacle_map
    obstacle_map = cv2.dilate(obstacle_map, kernel)

    kernel = np.ones((5, 5), dtype=np.uint8)
    # show_ex = cv2.inRange(explored_map,0.1,1) # old code, some times buggy
    show_ex = (explored_map == 1).astype(np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)
    contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(edge_map,contour,-1,1,1)

    # clear the boundary
    edge_map[0:2, 0:map_size]=0.0
    edge_map[map_size-2:map_size, 0:map_size-1]=0.0
    edge_map[0:map_size, 0:2]=0.0
    edge_map[0:map_size, map_size-2:map_size]=0.0

    target_edge = edge_map - obstacle_map

    target_edge[target_edge>0.8]=1.0
    target_edge[target_edge!=1.0]=0.0

    img_label, num = measure.label(target_edge, connectivity=2, return_num=True)
    props = measure.regionprops(img_label)

    # selem = skimage.morphology.disk(1)
    obstacle_map[current_pose[0], current_pose[1]] = 0
    selem = skimage.morphology.disk(1)
    traversible = np.logical_not(skimage.morphology.binary_dilation(
        dis_obstacle_map, selem))
    # traversible = 1 - traversible
    planner = FMMPlanner(traversible)
    goal_pose_map = np.zeros((obstacle_map.shape))
    goal_pose_map[current_pose[0], current_pose[1]] = 1
    planner.set_multi_goal(goal_pose_map)

    Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
    Goal_point = []
    Goal_cost_list = []
    dict_cost = {}
    for i in range(0, len(props)):

        if props[i].area > threshold_point:
            # dist = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])]
            # dict_cost[i] = props[i].area
            # print(dist)
            # print(props[i].area)
            dict_cost[i] = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])]
            # print(dict_cost[i])

    if dict_cost:
        dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=False)

        for i, (key, value) in enumerate(dict_cost):
            if value == planner.fmm_dist.max():
                continue
            Goal_edge[img_label == key + 1] = i + 1
            Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
            Goal_cost_list.append(value)
            if i == 5:
                break

    return  Goal_cost_list, Goal_edge, Goal_point


