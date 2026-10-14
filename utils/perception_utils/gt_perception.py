import numpy as np
import supervision as sv
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R

from constants import categories_21_plus_stairs

from utils.vln_logger import vln_logger
from utils.scene_graph_utils import habitat_to_agent_coord

def get_gt_goal_positions(goals_habitat, agent_state):
    # get the agent's camera position and rotation, this should be at T=0
    agent_pos = agent_state.sensor_states["depth"].position
    agent_rot = agent_state.sensor_states["depth"].rotation
    r = R.from_quat([agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w])
    rot = r.as_matrix()
    agent_to_world = np.eye(4)
    agent_to_world[:3, :3] = rot
    agent_to_world[:3, 3] = np.array(agent_pos)
    
    # get all habitat poase of the goals
    goal_positions = []
    for goal in goals_habitat:
        goal_positions.append(goal.position)
    goal_positions = np.array(goal_positions)
    
    # transform the goal positions to agent coordinates 
    goal_positions_agent = habitat_to_agent_coord(goal_positions, agent_to_world)
    
    return goal_positions_agent
    
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

def get_gt_bbox_and_caption(semantic_img, depth_img, semantic_annotations, category_list):
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
    class_ids = []
    
    semantic_img = semantic_img.astype(np.int32)
    depth_img = depth_img.astype(np.float32)
    depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())  
    
    # get categpry names from category ids
    category_mapping = {}
    if semantic_annotations is not None and len(semantic_annotations.objects) > 0:
        for obj in semantic_annotations.objects:
            if obj is not None and obj.category is not None:
                obj_numeric_id = int(obj.id.split("_")[-1])  
                category_mapping[obj_numeric_id] = obj.category.name()
    elif semantic_annotations is not None and len(semantic_annotations.categories) > 0:
        for obj in semantic_annotations.categories:
            obj_numeric_id = obj.index()
            category_mapping[obj_numeric_id] = obj.name()
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
        
        if mp3d_name not in categories_21_plus_stairs:
            continue # 02/05/2025 current pipeline only consider 21 categories
        
        if obj_mask.sum() < 50:
            continue # ignore when object too small?
        
        class_ids.append(np.argwhere(np.array(category_list) == mp3d_name)[0][0]) # get the index of the category in categories_21
        
        mask.append(obj_mask)
        xyxy.append(bbox)
        caption.append(mp3d_name)
        
    return np.array(mask), np.array(xyxy), np.ones((len(xyxy),)), caption, np.array(class_ids)

class GTPerception():
    def __init__(self):
        vln_logger.info("GTPerception initialized", extra={"module": "GTPerception"})
        self.set_category_list(categories_21_plus_stairs) # TODO: generalize to other datasets
        self.semantic_annotations = None
        
    def set_semantic_annotaions(self, semantic_annotations):
        self.semantic_annotations = semantic_annotations
        if self.semantic_annotations is None or (len(semantic_annotations.objects) == 0 and len(semantic_annotations.categories) == 0):
            vln_logger.info("No semantic annotations available.", extra={"module": "GTPerception"})
            self.semantic_annotations = None
    
    def set_category_list(self, category_list):
        # place holder to be generalize to other datasets
        self.category_list = category_list
        
    def detect(self, image_semantic, image_depth):
        if self.semantic_annotations is None:
            return None, None
        
        mask, xyxy, conf, caption, class_ids = get_gt_bbox_and_caption(image_semantic, image_depth, self.semantic_annotations, self.category_list)
        
        detections = None
        get_results = False
        
        if len(caption) > 0:
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=class_ids,
                mask=mask,
            )
        
            get_results = caption is not None
        else:
            detections = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty((0,)),
                class_id=np.empty((0,)),
                mask=np.empty((0,1,1))
            )
        
        return get_results, detections
    

