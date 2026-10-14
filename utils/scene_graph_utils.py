import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import torch
from utils.slam_classes import MapObjectList, DetectionList
import faiss
import cv2
import pandas as pd
import itertools
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import matplotlib as mpl
from matplotlib.path import Path

from utils.ious import compute_3d_iou_accuracte_batch, compute_iou_batch
from constants import categories_21, scene_to_room_labels
from utils.plotly_utils import *
from utils.vis import *
from utils.raycast import get_visible_unknown, seperate_map

def cluster_frontiers(frontiers, method="dbscan", num_clusters=None, eps=5, min_samples=2):
    """
    Cluster frontier locations in BEV space and return cluster centers using the most centric point.

    Parameters:
    - frontiers: (N,2) array of frontier coordinates.
    - method: "kmeans" or "dbscan".
    - num_clusters: Number of clusters (only for KMeans).
    - eps: Maximum distance for DBSCAN clustering.
    - min_samples: Minimum number of points for DBSCAN to form a cluster.

    Returns:
    - cluster_labels: List of cluster IDs for each frontier.
    - clusters: Dictionary mapping cluster_id → list of points.
    - cluster_centers: Dictionary mapping cluster_id → most centric (x, y).
    """
    frontiers = np.array(frontiers)  # Convert to NumPy array

    if method == "kmeans":
        if num_clusters is None:
            num_clusters = min(5, len(frontiers))  # Default: 5 clusters or less if N<5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(frontiers)
        cluster_centers = {}

        for i in range(num_clusters):
            cluster_points = frontiers[cluster_labels == i]
            if len(cluster_points) > 0:
                # Compute mean and find the closest actual point
                cluster_mean = np.mean(cluster_points, axis=0)
                closest_idx = np.argmin(np.linalg.norm(cluster_points - cluster_mean, axis=1))
                cluster_centers[i] = tuple(cluster_points[closest_idx])

    elif method == "dbscan":
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(frontiers)
        unique_labels = set(cluster_labels)

        cluster_centers = {}
        for label in unique_labels:
            if label != -1:  # Ignore noise points
                cluster_points = frontiers[cluster_labels == label]
                if len(cluster_points) > 0:
                    # Compute mean and find the closest actual point
                    cluster_mean = np.mean(cluster_points, axis=0)
                    closest_idx = np.argmin(np.linalg.norm(cluster_points - cluster_mean, axis=1))
                    cluster_centers[label] = tuple(cluster_points[closest_idx])

    # Group points by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tuple(frontiers[i]))

    return np.array(list(cluster_centers.values())).reshape(-1,2)

def merge_obj_image_feats(obj1, obj2):
    '''
    Merge the image features of the two objects from obj2 to obj1.
    Obj2 is the latest detection, the sliding window will keep the most recent 5 detections.
    Modification happens in place in obj1
    TODO: in the future, they could be more complicated fusion strategies
    '''
    obj1['image_feats'] = np.concatenate((obj1['image_feats'], obj2['image_feats']),axis=0)
    if obj1['image_feats'].shape[0] > 5:
        obj1['image_feats'] = obj1['image_feats'][-5:] # remove the earliest observations
    
    return obj1

def merge_obj_token_logits(obj1, obj2):
    '''
    Merge the image features of the two objects from obj2 to obj1.
    Obj2 is the latest detection, the sliding window will keep the most recent 5 detections.
    Modification happens in place in obj1
    TODO: in the future, they could be more complicated fusion strategies
    '''
    obj1['token_logits'] = np.concatenate((obj1['token_logits'], obj2['token_logits']), axis=0)
    if obj1['token_logits'].shape[0] > 5:
        obj1['token_logits'] = obj1['token_logits'][-5:] # remove the earliest observations
    return obj1

def compute_overlap_matrix_2set(cfg, objects_map: MapObjectList, objects_new: DetectionList) -> np.ndarray:
    '''
    compute pairwise overlapping between two set of objects in terms of point nearest neighbor. 
    objects_map is the existing objects in the map, objects_new is the new objects to be added to the map
    Suppose len(objects_map) = m, len(objects_new) = n
    Then we want to construct a matrix of size m x n, where the (i, j) entry is the ratio of points 
    in point cloud i that are within a distance threshold of any point in point cloud j.
    '''
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix = np.zeros((m, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map] # m arrays
    for i, points in enumerate(points_map):
        num_points = len(points)
        if num_points > cfg.max_num_points:
            choice = np.random.choice(range(num_points), cfg.max_num_points)
            points_map[i] = points[choice]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map] # m indices
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, points_map):
        index.add(arr)
        
    points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new] # n arrays
    for i, points in enumerate(points_new):
        num_points = len(points)
        if num_points > cfg.max_num_points:
            choice = np.random.choice(range(num_points), cfg.max_num_points)
            points_new[i] = points[choice]
        
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    except ValueError:
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        for pcd in objects_map.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        for pcd in objects_new.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        
        iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)
            

    # Compute the pairwise overlaps
    for i in range(m):
        for j in range(n):
            if iou[i,j] < 1e-6:
                continue
            
            D, I = indices[i].search(points_new[j], 1) # search new object j in map object i

            overlap = (D < cfg.downsample_voxel_size ** 2).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix

def compute_visual_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    # stack most recent image features TODO: naive method that ueses the last image features
    det_features = detection_list.get_stacked_values_torch('image_feats', -1)
    obj_features = objects.get_stacked_values_torch('image_feats', -1)
    
    if cfg.visual_sim_type == "clip":
        visual_sim = det_features @ obj_features.T
        norm_det = det_features.norm(dim=-1).reshape(-1, 1)
        norm_obj = obj_features.norm(dim=-1)
        visual_sim = visual_sim / norm_det / norm_obj
    else:
        raise ValueError(f"Invalid visual similarity type: {cfg.visual_sim_type}")
    
    return visual_sim

def text2value(text):
    try:
        value = float(text)
    except:
        value = 0
    return value

def calculate_entropy(prob_list):
    '''
    Calculate the entropy of a probability distribution
    
    Input: 
        - prob_list: a list of probabilities
    Ouput:
        - entropy_list: a list of the entropy of the probability distribution
    '''
    entropy_list = []
    for prob in prob_list:
        entropy = - (prob * np.log(prob)).sum()
        entropy_list.append(entropy)
    return entropy_list


def bounding_boxes_and_mask(semantic_img, depth_img):
    '''
    Input:
        - semantic_img: a 2D numpy array representing the semantic segmentation image
        - depth_img: a 2D numpy array representing the depth image
    Output:
        - obj_id_list: a list of object ids (int)
        - bbox_list: a list of bounding boxes ((x,y,x,y) tuples)
        - obj_mask_list: a list of object masks (np.array (H,W))
    '''
    
    unique_ids = np.unique(semantic_img)
    obj_id_list = []
    bbox_list = []
    obj_mask_list = []

    for obj_id in unique_ids:
        if obj_id == 0:
            continue

        mask = (semantic_img == obj_id).astype(np.uint8)
        depth_mask = (depth_img > 0).astype(np.uint8)  
        mask_expanded = cv2.bitwise_and(mask, depth_mask)  
        kernel = np.ones((5, 5), np.uint8)
        mask_expanded = cv2.dilate(mask_expanded, kernel, iterations=2)

        contours, _ = cv2.findContours(mask_expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            obj_mask = np.zeros_like(mask)
            obj_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
            
            # if object mask is too small, skip, this is a heuristic
            if obj_mask.sum() < 50:
                continue
            
            obj_id_list.append(obj_id)
            bbox_list.append((x, y, x + w, y + h))  
            obj_mask_list.append(obj_mask.squeeze(-1))
            
    return obj_id_list, bbox_list, obj_mask_list

def get_gt_bbox_and_caption(semantic_img, depth_img, semantic_annotations):
    '''
    Get the ground turth bounding boxes and captions from the semantic image and 
    depth image, hm3d category name is converted to mp3d category name, and filtered
    by the categories_21.
    
    Input:
        - semantic_img: a 2D numpy array representing the semantic segmentation image
        - depth_img: a 2D numpy array representing the depth image
        - semantic_annotations: a habitat-sim semantic annotation object
        
    Output:
        - mask: np.array (n,H,W)
        - xyxy: np.array (n,4)
        - conf: np.array (n,)
        - caption: list of strings of object labels
    '''
    if semantic_img is None:
        return None, None, None, None
    
    mask = []
    xyxy = []
    caption = []
    
    semantic_img = semantic_img.astype(np.int32)
    depth_img = depth_img.astype(np.float32)
    depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())  
    
    # get categpry names from category ids
    category_mapping = {}
    if semantic_annotations is not None and hasattr(semantic_annotations, "objects"):
        for obj in semantic_annotations.objects:
            if obj is not None and obj.category is not None:
                obj_numeric_id = int(obj.id.split("_")[-1])  
                category_mapping[obj_numeric_id] = obj.category.name()
    else:
        print("No semantic annotations available.")
    
    # adding each bbox and caption to the list
    obj_id_list, bbox_list, obj_mask_list = bounding_boxes_and_mask(semantic_img, depth_img)
    
    df = pd.read_csv('tools/matterport_category_mappings.tsv', sep='    ', header=0, engine='python') # somehow tsv is separated by 4 spaces not tab
    
    for obj_id, bbox, obj_mask in zip(obj_id_list, bbox_list, obj_mask_list):
        hm3d_name = category_mapping.get(obj_id, "Unknown") # hm3d cateogry name
        row = df[df['raw_category'] == hm3d_name]
        
        if len(row) != 1:
            continue # not mapped in the df
        
        mp3d_name = row['mpcat40'].values[0] # update cateogry name to mp3d name
        
        if mp3d_name not in categories_21:
            continue # 02/05/2025 current pipeline only consider 21 categories
        
        mask.append(obj_mask)
        xyxy.append(bbox)
        caption.append(mp3d_name)
        
    return np.array(mask), np.array(xyxy), np.ones((len(xyxy),)), caption 

def world_to_local_2d(world_pos, agent, meters_per_pixel):
    """
    Convert a 3D world point to the occupancy map's 2D frame:
        - Subtract the agent's world position.
        - Scale by meters_per_pixel (meters -> pixels).
        - Shift so that the agent is at the provided occupancy map coordinate.
    Returns [x_local, y_local] in pixel space.
    """

    agent_pos = agent["position"]
    agent_rot = agent["rotation"]
    # agent_rot = quaternion.as_rotation_matrix(agent.sensor_states['rgb'].rotation)
    relative = np.array(world_pos) - np.array(agent_pos)  # Translate
    # relative = np.array(world_pos + [1]) 

    rotation_world_start = quaternion_from_coeff(agent_rot)
    # agent_rot = vln_logger.sim.get_agent(0).scene_node.node_sensor_suite.get_sensors()['rgb'].node.transformation
    # world_point = np.dot(agent_rot.T, relative) 

    world_point = quaternion_rotate_vector(rotation_world_start.inverse(), relative)

    dx = int(400-world_point[2]/meters_per_pixel)
    dy = int(400-world_point[0]/meters_per_pixel)
    # world_point = np.dot(agent_rot.T, relative)  # Rotate using transpose
    return np.array([dx, dy])


def category_mp3d_mapping(hm3d_name, df):
    row = df[df['raw_category'] == hm3d_name]
        
    if len(row) == 1:
        mp3d_name = row['mpcat40'].values[0] # update cateogry name to mp3d name
        # if (mp3d_name not in categories_21):
        #     mp3d_name = None
    else:
        mp3d_name = None
    return mp3d_name


def region_overlap(region1, region2, min_overlap=0.5):
    """
    Determine whether two regions (with grid bbox coordinates) overlap significantly.
    The region_bbox here is a 2x2 array (min and max in grid x,z).
    
    Input:
        - region1, region2: dictionaries with "region_bbox" keys (numpy arrays).
        - min_overlap: minimum overlap ratio for significant overlap.
    Output:
        - True if the regions overlap significantly, False otherwise.
    """
    if region1["bbox"] is None or region2["bbox"] is None:
        return False  
    bbox1_min, bbox1_max = np.array(region1["bbox"])
    bbox2_min, bbox2_max = np.array(region2["bbox"])
    overlap_x = max(0, min(bbox1_max[0], bbox2_max[0]) - max(bbox1_min[0], bbox2_min[0]))
    overlap_y = max(0, min(bbox1_max[1], bbox2_max[1]) - max(bbox1_min[1], bbox2_min[1]))
    area1 = (bbox1_max[0] - bbox1_min[0]) * (bbox1_max[1] - bbox1_min[1])
    area2 = (bbox2_max[0] - bbox2_min[0]) * (bbox2_max[1] - bbox2_min[1])
    intersection_area = overlap_x * overlap_y
    if intersection_area == 0:
        return False
    iou = intersection_area / (area1 + area2 - intersection_area)
    return iou > min_overlap

def load_scene_graph(
    sim, 
    agent,    #  (400, 400)
    aligned_bbox=True,
    meters_per_pixel=0.05,
    distance_threshold=4.0,
    connected_thre=90
):
    """ 
    Since Habitat does not provide region centers or sizes, the region's 2D bounding
    box and center are computed from the objects within the region.
    """
    df = pd.read_csv('tools/matterport_category_mappings.tsv', sep='    ', header=0, engine='python') # somehow tsv is separated by 4 spaces not tab

    scene_graph = {"rooms": []}
    # nav mesh coordinate: world coordinate
    semantic_scene = sim.semantic_scene
    if semantic_scene is None:
        print("Error: No semantic scene available!")
        return scene_graph
    
    agent_state = sim.get_agent_state()
    cur_agent_pos = agent_state.position  # shape (3,) -> (x, y, z)
    agent_pos = agent["position"]

    # def world_to_local_2d(world_point):
    #     """
    #     Convert a 3D world point to the occupancy map's 2D frame:
    #       - Subtract the agent's world position.
    #       - Scale by meters_per_pixel (meters -> pixels).
    #       - Shift so that the agent is at the provided occupancy map coordinate.
    #     Returns [x_local, y_local] in pixel space.
    #     """
    #     dx = (world_point[0] - agent_pos[0]) / meters_per_pixel
    #     dy = (world_point[1] - agent_pos[1]) / meters_per_pixel
    #     return np.array([400 + dx, 400 - dy])

    def compute_3d_distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    objec = []
    all_regions = []
    for region in semantic_scene.regions:
        if region is None:
            continue
        region_id_str = region.id.split("_")[-1]
        try:
            region_id = int(region_id_str)
        except:
            continue
        if region_id < 0:
            continue

        kept_objects = []
        for obj in region.objects:
            if obj is None or obj.obb is None:
                continue
            object_id_str = obj.id.split("_")[-1]
            try:
                object_id = int(object_id_str)
            except:
                continue

            if aligned_bbox:
                center_world = obj.obb.center
                size_world = obj.obb.sizes
                rot_quat = [0, 0, 0, 1]
            else:
                center_world = obj.obb.center
                size_world = obj.obb.sizes if obj.obb is not None else obj.obb.sizes
                rot_quat = obj.obb.rotation[[1, 2, 3, 0]]
            size_world = np.array(size_world)
            center_local_2d = world_to_local_2d(center_world, agent, meters_per_pixel)
            
            # # occupancy range filter
            # if not (0 <= center_local_2d[0] < W and 0 <= center_local_2d[1] < H):
            #     continue  #filter1--skips objects that are ouside the map
            
        
            # # Skip objects that are too far from the agent.
            if compute_3d_distance(center_world, agent_pos) > distance_threshold:
                continue #filter1--skips objects that are ouside the threshold
            
            # # if you want only specific objects or the object and agent are not at the same floor
            hm3d_name = obj.category.name() if obj.category else "unknown"
            caption = category_mp3d_mapping(hm3d_name, df)
            # print(caption, cur_agent_pos[1], agent_pos[1], center_world[1])
            if caption is None or (abs(cur_agent_pos[1] - center_world[1]) > 2.0):
            # if (caption.lower() not in ['bed', 'picture', 'toilet']) or (abs(cur_agent_pos[1] - center_world[1]) > 2.0):
            # if (abs(cur_agent_pos[1] - center_world[1]) > 2.0) or (caption.lower() in ['wall', 'window frame', 'window', 'unknown']):
            # if caption.lower() not in object_caption:
                # print("Filtered: ", caption)
                continue
            
            size_grid = size_world / meters_per_pixel
            half_size_x = size_grid[0] / 2.0
            half_size_z = size_grid[1] / 2.0
            node_bbox_2d = np.array([
                [center_local_2d[0] - half_size_x, center_local_2d[1] - half_size_z],
                [center_local_2d[0] + half_size_x, center_local_2d[1] + half_size_z]
            ])
            if obj.category: 
                name = obj.category.name()
                objec.append(name)
            else: 
                name = "Unknown"
                
                
            object_node = {
                "id": f"{region_id}_{object_id}",
                "caption": obj.category.name() if obj.category else "Unknown",
                "bbox": node_bbox_2d.tolist(),
                "center": center_local_2d.tolist(),
                "rotation": rot_quat,
                "size": size_grid.tolist(),
            }
            kept_objects.append(object_node)

        #go on if the region has at least one valid object.
        if len(kept_objects) > 0:
           
            x_min = min(obj["bbox"][0][0] for obj in kept_objects)
            y_min = min(obj["bbox"][0][1] for obj in kept_objects)
            x_max = max(obj["bbox"][1][0] for obj in kept_objects)
            y_max = max(obj["bbox"][1][1] for obj in kept_objects)
            region_bbox_local_2d = np.array([[x_min, y_min], [x_max, y_max]])
            region_center_local_2d = region_bbox_local_2d.mean(axis=0)

            region_node = {
                "id": str(region_id),
                "bbox": region_bbox_local_2d.tolist(),
                "level_id": region.level.id if hasattr(region.level, "id") else "Unknown",
                "center": region_center_local_2d.tolist(),
                "connected_regions": [],
                "objects": kept_objects
            }
            all_regions.append(region_node)
    # print("name of all the objects in the scene:", objec)
    rooms_dict = {}
    for region_node in all_regions:
        room_id = region_node["level_id"]
        if room_id not in rooms_dict:
            rooms_dict[room_id] = {"id": room_id, "regions": []}
        rooms_dict[room_id]["regions"].append(region_node)

    # Connect overlapping or nearby regions.
    pruned_rooms = []
    for room_data in rooms_dict.values():
        if len(room_data["regions"]) == 0:
            continue
        regions_in_room = room_data["regions"]
        for i in range(len(regions_in_room)):
            for j in range(i + 1, len(regions_in_room)):
                r1 = regions_in_room[i]
                r2 = regions_in_room[j]
                if r1["bbox"] is None or r2["bbox"] is None:
                    continue
                if region_overlap(r1, r2):
                    r1["connected_regions"].append(r2["id"])
                    r2["connected_regions"].append(r1["id"])
                else:
                   
                    c1 = np.array(r1["center"])
                    c2 = np.array(r2["center"])
                    if np.linalg.norm(c1 - c2) < connected_thre:
                        r1["connected_regions"].append(r2["id"])
                        r2["connected_regions"].append(r1["id"])
        pruned_rooms.append(room_data)

    scene_graph["rooms"] = pruned_rooms
    return scene_graph

def habitat_to_agent_coord(habitat_coord, agent_to_world):
    '''
    Input:
        - habitat_coord: a 3D point in habitat coordinate (n, 3) numpy array
        - agent_to_world: the transformation matrix from agent to world (4, 4) transformation
    Output:
        - agent_coord: a 3D point in agent coordinate (n, 3) numpy array
    '''
    habitat_to_agent_coord_transform = np.eye(4)
    habitat_to_agent_coord_transform[:3, :3] = np.array([[0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]])
    
    relative_to_agent = np.linalg.inv(agent_to_world) @ np.hstack((habitat_coord, np.ones((habitat_coord.shape[0], 1)))).T
    agent_coord = habitat_to_agent_coord_transform @ relative_to_agent
    return agent_coord[:3, :].T
    
def get_gt_scenegraph(
    scene_name,
    semantic_scene,
    # sim, 
    agent_state, 
    map_resolution,
    origins_grid,
):   
    # nav mesh coordinate: world coordinate
    if semantic_scene is None:
        print("Error: No semantic scene available!")
        return scene_graph
    
    try:
        id_to_room_labels = scene_to_room_labels[scene_name]
    except:
        print("Error: No room labels available for this scene!")
        id_to_room_labels = ["UNKNOWN"] * 50 # haven't specify the room name for this scene
    
    # mp3d category mapping
    df = pd.read_csv('tools/matterport_category_mappings.tsv', sep='    ', header=0, engine='python') # somehow tsv is separated by 4 spaces not tab
    
    # get the agent's camera position and rotation, this should be at T=0
    agent_pos = agent_state.sensor_states["depth"].position
    agent_rot = agent_state.sensor_states["depth"].rotation
    r = R.from_quat([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
    rot = r.as_matrix()
    agent_to_world = np.eye(4)
    agent_to_world[:3, :3] = rot
    agent_to_world[:3, 3] = np.array(agent_pos)
    
    scene_graph = {"floors": [{}]} # assume only one floor now

    def get_all_bbox_points(obj):
            all_points = []
            # use objecct center and half extents to get all 8 points
            center = np.array(obj.obb.center)
            half_extents = np.array(obj.obb.half_extents)
            for signs in itertools.product([-1, 1], repeat=3):
                all_points.append(center + half_extents * np.array(signs))
            return all_points
    
    def get_bbox_2d_agent_coord(all_3d_bbox_points):
        all_3d_bbox_points = np.array(all_3d_bbox_points) # (n, 3)
        bbox_3d_agent_coord = habitat_to_agent_coord(all_3d_bbox_points, agent_to_world) # (n, 3)
        
        # get min and max points in terms of x and z
        x_min_point = bbox_3d_agent_coord[np.argmin(bbox_3d_agent_coord[:, 0])].tolist()
        x_max_point = bbox_3d_agent_coord[np.argmax(bbox_3d_agent_coord[:, 0])].tolist()
        z_min_point = bbox_3d_agent_coord[np.argmin(bbox_3d_agent_coord[:, 2])].tolist()
        z_max_point = bbox_3d_agent_coord[np.argmax(bbox_3d_agent_coord[:, 2])].tolist()
        
        # get minimum-area bounding rectangle 
        points = np.vstack((bbox_3d_agent_coord[:, 0], bbox_3d_agent_coord[:, 2])).T.astype(np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        
        # get min max value to indicate region height
        y_min = np.min(bbox_3d_agent_coord[:, 1])
        y_max = np.max(bbox_3d_agent_coord[:, 1])
        # get region center
        x_center = (x_min_point[0] + x_max_point[0]) / 2
        z_center = (z_min_point[2] + z_max_point[2]) / 2
        y_center = (y_min + y_max) / 2
        region_center = [x_center, y_center, z_center]
        y_avg = bbox_3d_agent_coord[:, 1].mean()
        
        # NOTE: insert y value to box to maintain consistency for now
        new_column = np.ones((box.shape[0], )) * y_avg
        box = np.insert(box, 1, new_column, axis=1) # insert y value to box
        
        return box.tolist(), y_min, y_max, y_avg,region_center
        # return (x_min_point, z_min_point, x_max_point, z_max_point), y_min, y_max, y_avg,region_center
    
    ##### debug print all valid region lables #####
    # temp_list = []
    # for region in semantic_scene.regions:
    #     region_id = int(region.id.split("_")[-1])
    #     temp_list.append(region_id)
    # print("all ids:", temp_list)
    # pass
    
    ##### adding regions #####
    sg_region_list = []
    region_floor_heights =[]
    for region in semantic_scene.regions:
        sg_region = {'objects': []}
        # check if valid region
        if region is None:
            continue
        region_id_str = region.id.split("_")[-1]
        try:
            region_id = int(region_id_str)
        except:
            continue
        if region_id < 0:
            continue
        
        ##### adding objects #####
        all_3d_bbox_points = []
        for obj in region.objects:
            # check if valid object
            if obj is None or obj.aabb is None:
                continue
            object_id_str = obj.id.split("_")[-1]
            try:
                object_id = int(object_id_str)
            except:
                continue
            object_name = obj.id.split("_")[0]
            if object_name in ["wall", "ceiling", "floor", "Uknown"]:
                continue
            
            caption = category_mp3d_mapping(object_name, df)
            if caption is None:
                continue
            
            # append all 8 bbox defined points to the list for post processing
            all_3d_bbox_points += get_all_bbox_points(obj)
            obj_agent_coord = habitat_to_agent_coord(np.array(obj.obb.center).reshape(1,3), agent_to_world).reshape(3,)   
            sg_obj = {
                'caption': caption,
                'id': object_id,
                'region_id': region_id,
                'center': [int(np.floor(obj_agent_coord[0]*100 / map_resolution) + origins_grid[0]), 
                                int(np.floor(obj_agent_coord[2]*100 / map_resolution) + origins_grid[1])],
                'confidence': 1.0,
                'center_mentric': obj_agent_coord.tolist(),
            }
            if caption not in categories_21:
                continue # don't include objects that we don't care about
            sg_region['objects'].append(sg_obj)
        ##### adding objects #####
        
        if len(all_3d_bbox_points) == 0:
            continue # skip if there's no object in the region
        
        ##### region post processing #####
        bbox, y_min, y_max, y_avg, region_center = get_bbox_2d_agent_coord(all_3d_bbox_points)
        region_name = id_to_room_labels[region_id]
        sg_region['caption'] = region_name
        sg_region['confidence'] = 1.0
        sg_region['center'] = [int(np.floor(region_center[0]*100 / map_resolution) + origins_grid[0]),
                              int(np.floor(region_center[2]*100 / map_resolution) + origins_grid[1])]
        sg_region['id'] = region_id
        sg_region['bbox'] = bbox # this is in metric space
        sg_region['height_min_metric'] = y_min
        sg_region['height_max_metric'] = y_max
        sg_region['height_avg_metric'] = y_avg
        sg_region['center_metric'] = region_center
        sg_region_list.append(sg_region)
        # region_floor_heights.append(region_center[1])
        region_floor_heights.append(sg_region['height_avg_metric'] )
    ##### adding regions #####
    
    ##### post processing of scene graph #####
    from sklearn.cluster import AgglomerativeClustering
    # use DBSCAN to do floor clustering 
    X = np.array(region_floor_heights).reshape(-1,1)
    # print(X)
    # db = DBSCAN(eps=1, min_samples=1).fit(X)
    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=1).fit(X)
    # ms = MeanShift().fit(X)
    # labels = db.labels_
    cluster_labels = ac.labels_

    # print(np.unique(db.labels_))
    # print(np.unique(ac.labels_))
    # print(np.unique(ms.labels_))

    # use cluster to assign floor 
    floor_heights = [np.mean(np.array(region_floor_heights)[np.where(cluster_labels == i)]) for i in np.unique(cluster_labels)]
    floor_low_to_hight = np.argsort(np.array(floor_heights), axis=0).reshape(-1)
    scene_graph["floors"] = []
    for i, indx in enumerate(floor_low_to_hight):
        cluster_label = np.unique(cluster_labels)[indx]
        floor_dict = {
            "floor_id": int(i),
            "regions": [],
            "floor_avg_height": np.mean(np.array(region_floor_heights)[np.where(cluster_labels == cluster_label)]),
        }
        scene_graph["floors"].append(floor_dict)
        
    for index in range(len(sg_region_list)):
        cluster_label = cluster_labels[index]
        region_instance = sg_region_list[index]
        floor_id = np.where(floor_low_to_hight == cluster_label)[0][0]
        region_instance['floor_id'] = int(floor_id)
        for object in region_instance['objects']:
            object['floor_id'] = int(floor_id)
        scene_graph["floors"][floor_id]['regions'].append(region_instance)
    
    # import json
    # with open("scene_graph.json", "w") as f:
    #     json.dump(scene_graph, f, indent=4)
        
    # debug 
    # scene_graph["floors"][0]['regions'] = sg_region_list
    
    return scene_graph

def shift_gt_sg(scenegraph, shift, axis):
    '''
    Shift the scene graph by a certain amount in the x or z direction.
    Input:
        - scenegraph: the scene graph to be shifted
        - shift: the amount to shift
        - axis: the axis to shift along (0 for x, 1 for z)
    Output:
        - scenegraph: the shifted scene graph
    '''
    # Shift the regions centers
    for floor in scenegraph["floors"]:
        for region in floor["regions"]:
            region["center"][axis] += shift
            # Shift the object centers
            for obj in region["objects"]:
                obj["center"][axis] += shift
    return scenegraph


def xyz_to_grid(xyz, grid_size, origins_grid):
    return np.array([
        np.floor((xyz[:, 0])*100 / grid_size).astype(int) + int(origins_grid[0]),
        np.floor((xyz[:, 2])*100 / grid_size).astype(int) + int(origins_grid[1]),
    ]).T

def update_region(region, grid_size, origins_grid):
    if 'bbox' in region:
        # gt
        bbox = xyz_to_grid(np.array(region['bbox']), grid_size, origins_grid)
        region['grid_bbox'] = bbox.tolist()
    else:
        # obs
        object_centers = []
        for obj in region.get('objects'):
            object_centers.append(obj['center'])
        if len(object_centers) == 0:
            object_centers = region['center']
        object_centers = np.array(object_centers).astype(np.int32).reshape(-1, 2)
        rect = cv2.minAreaRect(object_centers)
        bbox = np.array(cv2.boxPoints(rect))
        region['grid_bbox'] = bbox.tolist()
        region['center'] = list(rect[0])
        if not 'caption' in region and len(region['objects']) == 0:
            region['caption'] = 'unknown'
    return region

def get_nearby_regions(positions, regions, knn=3, distance=80):
    tree = KDTree([x['center'] for x in regions])
    knn = min(knn, len(regions))
    nearby_regions = []
    for position in positions:
        region_list = []
        distances, indices = tree.query(position, k=knn)
        if not isinstance(distances, np.ndarray):
            distances, indices = np.array([distances]), np.array([indices])
        for dist, index in zip(distances, indices):
            is_predicted = regions[index].get("predicted", False)
            if isinstance(regions[index]['caption'], dict):
                caption_list = list(regions[index]['caption'].keys())
                conf_list = list(regions[index]['caption'].values())
                is_confident = np.max(conf_list) > 0.2 and caption_list[np.argmax(conf_list)] != 'unknown'
            else:
                is_confident = regions[index]['caption'] != 'unknown'
            if dist <= distance and is_confident and not is_predicted:
                region_list.append(regions[index])
        nearby_regions.append(region_list)
    return nearby_regions

def plot_region(img, regions, map_size, flip=False, min_obj_num=0, plot_objects=True):
    if len(regions) == 0:
        return img
    viridis = mpl.colormaps['gist_rainbow'].resampled(len(regions))
    colors = (viridis(range(len(regions)))[:, :3] * 255).astype(np.uint8)
    for i, region in enumerate(regions):
        if plot_objects:
            for obj in region.get('objects'):
                if flip:
                    obj_center = [int(obj['center'][1]), int(obj['center'][0])]
            else:
                obj_center = [int(obj['center'][1]), int(map_size-obj['center'][0])]
            try:
                cv2.circle(img, obj_center, 4, (200, 200, 0), 2)
            except Exception as e:
                print(f"Error plotting object center: {e}")
                print(f"Object center: {obj_center}")
                print(f"Image shape: {img.shape}")
                continue
        if len(region.get('objects')) < min_obj_num and not region.get("predicted", False):
            continue
        pts = np.array(region['grid_bbox'])
        pts = pts.reshape((-1, 2)).astype(np.int32)
        if flip:
            pts[...,0], pts[...,1] = pts[...,1], pts[...,0]
            region_center = [int(region['center'][1]), int(region['center'][0])]
        else:
            pts[...,0], pts[...,1] = pts[...,1], map_size - pts[...,0]
            region_center = [int(region['center'][1]), int(map_size-region['center'][0])]
        cv2.polylines(img, [pts], isClosed=True, color=colors[i].tolist(), thickness=2)
        img = cv2.circle(img, region_center, 3, (0, 255, 0), -1)
        if isinstance(region['caption'], dict):
            captions, confidences = get_sorted_key_value_lists(region['caption'])
            if region.get("predicted", False):
                region_caption = f'{region["id"]}: {captions[0]}'
            else:
                region_caption = captions[0]
        else:
            if region.get("predicted", False):
                region_caption = f'{region["id"]}: {region["caption"]}'
            else:
                region_caption = region['caption']
        if region.get("predicted", False):
            text_color = (200,50,50)
        else:
            text_color = (50,50,50)
        img, _ = add_text(img, f"{region_caption}", region_center, font_scale=0.35, color=text_color, thickness=1, horizontal_align='center', vertical_align='center')
    return img

# def plot_region(img, regions, map_size):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     fig.patch.set_facecolor('white')
#     ax.set_facecolor('white')

#     ax.imshow(img)
#     ax.set_aspect('equal')
#     ax.axis('off')

#     # Setup colors
#     viridis = mpl.colormaps['gist_rainbow'].resampled(len(regions))
#     colors = viridis(range(len(regions)))[:, :3]

#     for i, region in enumerate(regions):
#         # Convert bbox coordinates
#         pts = region['grid_bbox'].reshape((-1, 2)).astype(np.float32)
#         pts[:, 0], pts[:, 1] = pts[:, 1], map_size - pts[:, 0]  # Swap & flip Y

#         # Close the loop
#         pts = np.vstack([pts, pts[0]])

#         # Plot polygon in XY plane at Z=1
#         ax.plot(pts[:, 0], pts[:, 1], zs=1, zdir='z', color=colors[i], linewidth=2)

#         # Region center
#         center = np.array([region['center'][1], map_size - region['center'][0]])
#         ax.scatter(center[0], center[1], 1, color='green', s=20)

#         # Annotate
#         label = f"({region['id']}: {region['caption']})"
#         ax.text(center[0], center[1], 1.5, label, fontsize=6, ha='center')

#     # Clean up axes
#     ax.set_zlim(0, 2)
#     ax.view_init(elev=90, azim=-90)
#     ax.axis('off')

#     plt.tight_layout()
#     return fig

def check_overlap(region, matched_region, relaxed=False):
    # if a[center] is in b[bbox] and b[center] is in a[bbox]
    polygon_a = Path(region['grid_bbox'])
    polygon_b = Path(matched_region['grid_bbox'])
    is_a_in_b = polygon_b.contains_point(region['center'])
    is_b_in_a = polygon_a.contains_point(matched_region['center'])
    if relaxed:
        return is_a_in_b or is_b_in_a
    else:
        return is_a_in_b and is_b_in_a

def get_sorted_key_value_lists(data):
    """
    Returns two separate lists: keys and their corresponding values,
    ordered by value in descending order.

    Args:
        data (dict): Dictionary with string keys and numeric values

    Returns:
        tuple: (list of keys, list of corresponding values)
    """
    if not isinstance(data, dict):
        return [], []
    try:
        sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    except Exception as e:
        print("Error in get_sorted_key_value_lists: ", e)
        return [], []
    keys = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    return keys, values

def get_text_sim_score(clip_model_list, obs, target, topk=2):
    def get_pair_score(obs, target):
        obs_feat = get_text_feat(clip_model_list, obs)
        target_feat = get_text_feat(clip_model_list, target)
        sim_score = obs_feat @ target_feat.T
        return sim_score.item()
    if isinstance(target, dict):
        target, obs = obs, target
    if isinstance(obs, dict):
        captions, confidences = get_sorted_key_value_lists(obs)
        captions, confidences = captions[:topk], confidences[:topk]
        sim_scores = [get_pair_score(caption, target)*confidences[i] for i, caption in enumerate(captions)]
        if len(sim_scores)==0:
            sim_score = 0
        else:
            sim_score = np.sum(sim_scores)
    else:
        sim_score = get_pair_score(obs, target)
        captions = {obs: 1}
        sim_scores = [sim_score]
    return sim_score, captions, sim_scores

def get_text_feat(clip_model_list, text):
    clip_model, clip_preprocess, clip_tokenizer = clip_model_list
    with torch.no_grad():
        # TODO: debug text too long, it gives the all thought process instead of a caption
        tokenized_text = clip_tokenizer([text[:77]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    return text_feat

def detect_match(keys, queries, clip_model_list, knn=None, distance=None, overlap_relaxed=False, corr_score=None, topk=2, unique_match=True):
    """
    set either knn or distance or corr_thresh to filter matches
    overlap_relaxed: True/False/None, None means no check for region match
    """
    if knn is None:
        knn = len(keys)
    else:
        knn = min(knn, len(keys))
    matches = []
    if len(keys) == 0 or len(queries) == 0:
        return matches, 0
    tree = KDTree([x['center'] for x in keys])
    for query in queries:
        distances, indices = tree.query(query['center'], k=knn)
        if not isinstance(distances, np.ndarray):
            distances, indices = np.array([distances]), np.array([indices])
        matches += [dict(k=keys[i], q=query, dist=dist) for i, dist in zip(indices, distances)]
    if distance is not None:
        matches = [*filter(lambda x: x['dist'] <= distance, matches)]
    if overlap_relaxed is not None:
        matches = [*filter(lambda x: check_overlap(x['k'], x['q'], relaxed=overlap_relaxed), matches)]
    if corr_score is not None:
        # calculate corr_score
        for match in matches:
            if isinstance(match['k']['caption'], dict):
                match['corr_score'], _, match['sim_scores'] = get_text_sim_score(clip_model_list, match['k']['caption'], match['q']['caption'], topk=topk)
            else:
                match['corr_score'], _, match['sim_scores'] = get_text_sim_score(clip_model_list, match['q']['caption'], match['k']['caption'], topk=topk)
        matches = [*filter(lambda x: x['corr_score'] >= corr_score, matches)]
    else:
        matches = sorted(matches, key=lambda x: x['dist'], reverse=False)
    if unique_match:
        # keep only one match for each query according to corr_score
        if corr_score is not None:
            matches = sorted(matches, key=lambda x: x['corr_score'], reverse=True)
        else:
            matches = sorted(matches, key=lambda x: x['dist'], reverse=False)
        seen = set()
        new_matches = []
        for match in matches:
            if match['q']['id'] in seen:
                continue
            seen.add(match['q']['id'])
            new_matches.append(match)
        matches = new_matches
    if corr_score is not None:
        corr_scores = [match['corr_score'] for match in matches]
        score = sum(corr_scores) / (len(queries)+1e-6)
    else:
        score = len(matches) / (len(queries)+1e-6)
    return matches, score

def plot_matches(k_vis, q_vis, matches, reversed=False, map_size=480):
    vis = np.concatenate((k_vis, q_vis), axis=1)
    for match in matches:
        k_center = np.array([match['k']['center'][1], map_size-match['k']['center'][0]]).astype(np.int32)
        q_center = np.array([match['q']['center'][1], map_size-match['q']['center'][0]]).astype(np.int32)
        if reversed:
            k_center += np.array([map_size, 0])
        else:
            q_center += np.array([map_size, 0])
        cv2.circle(vis, k_center, 5, (255, 0, 0), 2)
        cv2.circle(vis, q_center, 5, (255, 0, 0), 2)
        cv2.line(vis, k_center, q_center, (255, 0, 0), 2)
        add_text(vis, f"{match['corr_score']:.2f}", (k_center+q_center)//2, font_scale=0.7, color=(200,200,0), thickness=2, horizontal_align='center', vertical_align='center')
    return vis

def crop_minAreaRect(img, rect):
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def check_visibility(center_i, center_j, wall_map):
    rect = cv2.minAreaRect(np.array([center_i, center_j]).astype(np.int32))
    rect = (rect[0], (max(rect[1][0], 2), max(rect[1][1], 10)), rect[2])
    wall = crop_minAreaRect(wall_map.T, rect)
    wall_pixels = wall.sum()
    return wall_pixels

def plot_visible_edge(fig, center_i, center_j, text, map_size=480):
    if fig is None:
        return
    fig.add_trace(
        go.Scatter(
            x=[center_i[1], center_j[1]],
            y=[map_size-center_i[0], map_size-center_j[0]],
            mode='markers+lines',
            marker=dict(size=6, color='red'),
            line=dict(color='orange', width=3),
            text=text,
        )
    )
def plot_object(fig, center_i, text, map_size=480):
    if fig is None:
        return
    fig.add_trace(
        go.Scatter(
            x=[center_i[1]],
            y=[map_size-center_i[0]],
            mode='markers',
            marker=dict(size=6, color='blue'),
            line=dict(color='pink', width=3),
            text=text,
        )
    )

class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank   = {i: 0 for i in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def detect_regions(objects, wall_map, knn_object=10, distance_threshold=30, wall_threshold=15, min_num=0, vis_fig=None, pred_sg=None):
    # 1) Build id->object map and collect all ids
    obj_map = {obj['id']: obj for obj in objects}
    ids = list(obj_map.keys())

    # 2) Initialize Union-Find on all ids
    uf = UnionFind(ids)

    # 3) For each object, query its nearby neighbors
    object_tree = KDTree([x['center'] for x in objects])
    knn_object = min(knn_object, len(objects))
    for id_i in ids:
        obj_i = obj_map[id_i]
        # look for all neighbors
        distances, indices = object_tree.query(obj_i['center'], k=knn_object)
        if not isinstance(distances, np.ndarray):
            distances, indices = np.array([distances]), np.array([indices])
        else:
            distances, indices = distances[:len(objects)], indices[:len(objects)]
        for dist, j in zip(list(distances), list(indices)):
            if dist > distance_threshold:
                continue
            obj_j = objects[j]
            if pred_sg is not None: 
                corr_score = pred_sg.get_text_sim_score(obj_i['caption'], obj_j['caption'])
                if corr_score < 0.5:
                    continue
            id_j = obj_j['id']
            center_i = obj_i['center']
            center_j = obj_j['center']
            wall_pixels = check_visibility(center_i, center_j, wall_map)
            if wall_pixels<=wall_threshold:
                uf.union(id_i, id_j)
                plot_visible_edge(vis_fig, center_i, center_j, f'{obj_i["caption"]} - {obj_j["caption"]}\nwall_pixels={wall_pixels}', map_size=wall_map.shape[0])
            else:
                plot_object(vis_fig, center_i, f'{obj_i["caption"]}\nwall_pixels={wall_pixels}', map_size=wall_map.shape[0])
    # 4) Bucket by root parent to form regions
    groups = {}
    for id in ids:
        root = uf.find(id)
        groups.setdefault(root,[]).append(obj_map[id])
    sg_regions = []
    for i, (_, objects) in enumerate(groups.items()):
        if min_num > 0 and len(objects) < min_num:
            continue
        sg_regions.append({
            'id': i,
            'objects': objects,
            'center': np.mean([obj['center'] for obj in objects], axis=0).tolist(),
            'caption': 'unknown',
            'corr_score': 0.0,
        })
    return sg_regions

def merge_regions(regions, clip_model_list, grid_size, origins_grid):
    region_map = {region['id']: region for region in regions}
    ids = list(region_map.keys())
    uf = UnionFind(ids)
    # match regions
    matches, score = detect_match(regions, regions, clip_model_list, knn=6, overlap_relaxed=None,unique_match=False)
    for match in matches:
        if match['k']['id'] == match['q']['id']:
            continue
        if isinstance(match['k']['caption'], dict):
            caption_k = max(match['k']['caption'].keys(), key=lambda x: match['k']['caption'][x])
            caption_q = max(match['q']['caption'].keys(), key=lambda x: match['q']['caption'][x])
            if caption_k == caption_q:
                uf.union(match['k']['id'], match['q']['id'])
        else:
            if match['k']['caption'] == match['q']['caption']:
                uf.union(match['k']['id'], match['q']['id'])
    # merge regions
    groups = {}
    for id in ids:
        root = uf.find(id)
        groups.setdefault(root,[]).append(region_map[id])
    new_regions = []
    for i, (_, matched_regions) in enumerate(groups.items()):
        if len(matched_regions)==1:
            new_regions.append(copy.deepcopy(matched_regions[0]))
        else:
            merged_region = copy.deepcopy(matched_regions[0])
            merged_caption = {}
            for region in matched_regions[1:]:
                merged_region['objects'] += region['objects']
                if isinstance(region['caption'], str):
                    merged_caption.setdefault(region['caption'], []).append(1)
                elif isinstance(region['caption'], dict):
                    for caption, conf in region['caption'].items():
                        merged_caption.setdefault(caption, []).append(conf)
            for caption, conf in merged_caption.items():
                merged_caption[caption] = sum(conf) / len(conf)
            merged_region['caption'] = merged_caption
            new_regions.append(merged_region)
    # update properties
    for region in new_regions:
        update_region(region, grid_size, origins_grid)
    return new_regions

def get_mask_center(mask):
    """
    Compute the center (centroid) of a binary mask.
    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s or 0s and 255s)
        
    Returns:
        (y, x): Tuple of center coordinates
    """
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Get coordinates of non-zero pixels
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None  # Empty mask

    center_y = np.mean(ys)
    center_x = np.mean(xs)
    return [center_y, center_x]

def get_unknown_cluster_centers(unknown_centers, eps=20, min_samples=1):
    """
    Group unknown centers into nearby regions and return the center of each cluster.

    Args:
        unknown_centers (np.ndarray): Array of shape (N, 2) with (y, x) coordinates.
        eps (float): Distance threshold for clustering.
        min_samples (int): Minimum number of points to form a cluster.

    Returns:
        list of tuple: List of (y, x) coordinates representing each cluster's center.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(unknown_centers)
    labels = clustering.labels_

    cluster_centers = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # skip noise
        cluster_points = unknown_centers[labels == cluster_id]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center.astype(int))
    cluster_centers = np.array(cluster_centers).reshape(-1, 2)
    return cluster_centers

def frontier_to_unknown(frontier_locations, occupancy_map, radius=64):
    unknown = (occupancy_map == 0) * 1.0
    obstacle = (occupancy_map == 1) * 1.0
    free = (occupancy_map == 2) * 1.0
    unknown, obstacle, free = seperate_map(occupancy_map)
    unknown_centers = []
    count = 1.0 * (radius*2) ** 2
    unknown_grids = []
    for i, loc in enumerate(frontier_locations):
        mask = get_visible_unknown(loc,600,obstacle,unknown,radius)
        center = get_mask_center(mask)
        if center is None:
            continue
        unknown_centers.append(center)
        unknown_grids.append(np.count_nonzero(mask)/count)
    
    unknown_idx = [i for i, count in enumerate(unknown_grids) if count >= 0.05]
    if len(unknown_idx) == 0:
        return []
    unknown_centers = np.array(unknown_centers)
    if len(unknown_idx) > 1:
        unknown_centers = get_unknown_cluster_centers(unknown_centers[unknown_idx])
    else:
        unknown_centers = unknown_centers[unknown_idx]
    return unknown_centers