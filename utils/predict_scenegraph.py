import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import json
import os
import clip
import torch
from copy import deepcopy
from utils.utils_llm import construct_query, query_llm, format_graph, format_response, load_json, text2value
from utils.scene_graph_utils import load_scene_graph, get_text_sim_score, detect_regions, update_region, detect_match, merge_regions, get_nearby_regions
from utils.vis_scenegraph import visualize_BEV
from utils.vis import remove_image_border
from utils.mapping import vote_class_name
from utils.slam_classes import MapObjectList
import copy
from utils.vln_logger import vln_logger
from PIL import Image
from constants import categories_21, region_captions, categories_21_plus_stairs, chance_regions
from collections import Counter
import cv2
def run_visualization(sub_obs_path, sub_pred_path, obs_path, pred_path, map_path, target, sg_img_save_path, agent_coordinate):
    from utils.vis_scenegraph import visualize_scenegraphs
    visualize_scenegraphs(
        sub_obs_path, sub_pred_path, obs_path, pred_path, map_path,
        str(target[0]), str(target[1]), sg_img_save_path, 
        str(agent_coordinate[0]), str(agent_coordinate[1])
    )

class PredictSceneGraph():
    def __init__(self, target, memory=None, clip_model_list=None, json_file=None, object_correlation='llm', region_correlation='text', caption_method='clip'):
        """use_text_score is for objects, region_correlation is for regions"""
        if clip_model_list is None:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
            self.clip_model.to(device='cuda')
            self.clip_tokenizer = clip.tokenize
            self.clip_model_list = (self.clip_model, self.clip_preprocess, self.clip_tokenizer)
        else:
            self.clip_model_list = clip_model_list
            self.clip_model, self.clip_preprocess, self.clip_tokenizer = clip_model_list
        self.clip_textual_cache = {}
        self.clip_visual_cache = {}
        self.object_correlation = object_correlation
        self.region_correlation = region_correlation
        self.caption_method = caption_method
        self.target = target
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompts = load_json(os.path.join(current_dir, 'phase.json'))
        self.object_list = MapObjectList() # used for offline analysis
        self.reset()
        # load object correlation
        if self.object_correlation == 'llm':
            corr_matrix_obj_llm = np.load(os.path.join(current_dir, '../tools/obj.npy'))
            corr_matrix_obj_llm -= corr_matrix_obj_llm.min()
            corr_matrix_obj_llm /= corr_matrix_obj_llm.max()
            self.corr_matrix_obj_llm = corr_matrix_obj_llm
        self.step = -1
        self.region_update_step = -1
        if self.object_correlation == 'cooccurrence':
            self.heatmap_obj_obj = np.load(os.path.join(current_dir, '../tools/heatmap_obj_obj.npy'), allow_pickle=True)
        if self.region_correlation == 'cooccurrence':
            self.heatmap_region_obj = np.load(os.path.join(current_dir, '../tools/heatmap_region_obj.npy'), allow_pickle=True)
        # import
        if memory is not None:
            room_nodes, group_nodes, nodes = self.init_from_sg(memory, target)
            scene_graph = {'rooms': room_nodes}
            self.scene_graph = self.merge_objects(scene_graph, distance_threshold=20.0, min_samples=1)
        if json_file is not None:
            self.init_from_json(json_file)
    
    def reset(self):
        for obj in self.object_list:
            if 'node' in obj:
                del obj['node']
        self.scene_graph = None
        self.object_nodes = []
        self.origins_grid = None
        self.step = -1
        self.region_update_step = -1
        self.object_id_count = 0
        self.obs_bev_image = None
        self.pred_bev_image = None
        self.predicted_sg = {}
        self.obs_sg = {}

    def set_target(self, target):
        if target == 'couch':
            target = 'sofa'
        elif target == 'tv':
            target = 'tv_monitor'
        self.target = target

    def set_step(self, step):
        self.step = step

    def move_grid_origin(self, shift, axis):
        for obj_node in self.object_nodes:
            obj_node["center"][axis] += shift
    def update(self, objects, grid_size, origins_grid, gradient_map, occupancy_map, image_history=None, llm_name='qwen2.5'):
        if len(objects) == 0:
            return
        self.origins_grid = origins_grid
        self.grid_size = grid_size
        vote_class_name(objects)
        new_object_list = [obj for obj in objects if not 'node' in obj]
        self.object_list.extend(new_object_list)
        new_object_nodes = [self.init_obj_node(obj, self.target) for obj in objects if not 'node' in obj]
        self.object_nodes.extend(new_object_nodes)
        new_scene_graph, num_region = self.group_objects(self.object_nodes, grid_size, origins_grid, gradient_map)
        new_scene_graph = self.assign_ids(new_scene_graph)
        if num_region==0:
            self.scene_graph = new_scene_graph
            return

        no_caption_sg = new_scene_graph
        if self.scene_graph is not None:
            no_caption_sg = self.match_region(new_scene_graph)

        if no_caption_sg is not None and self.caption_method != 'off':
            if self.caption_method == 'clip':
                new_caption_sg = self.generate_region_caption_TopK_clip(no_caption_sg, image_history, topk=2)
            elif self.caption_method == 'llm':
                new_caption_sg = self.generate_region_caption_TopK(no_caption_sg, self.target, llm_name=llm_name)

            if new_caption_sg is not None:
                for i, region in enumerate(new_caption_sg["rooms"][0]["regions"]):
                    no_caption_sg["rooms"][0]["regions"][i]['caption'] = region['caption']

        self.obs_sg = new_scene_graph
        if self.predicted_sg != {}:
            for room in self.predicted_sg.get("rooms", []):
                for region in room.get("regions", []):
                    update_region(region, grid_size, origins_grid)
            
            unexplored_pred_regions = []
            pred_regions = [region for room in self.predicted_sg.get("rooms", []) for region in room.get("regions", [])]
            for region in pred_regions:
                i, j = region['center']
                i, j = int(i), int(j)
                if i<0 or j<0 or i>=occupancy_map.shape[0] or j>=occupancy_map.shape[1]:
                    continue
                if occupancy_map[i, j] == 0:
                    unexplored_pred_regions.append(region)
            unexplored_pred_sg = {"rooms": [{"regions": unexplored_pred_regions, 'id': new_scene_graph['rooms'][0]['id']}]}
            new_scene_graph = self.merge_subgraph(self.obs_sg, unexplored_pred_sg)

        self.scene_graph = new_scene_graph
    
    def match_region(self, new_scene_graph):
        # match new region with old region and keep the caption
        old_regions = [region for room in self.scene_graph['rooms'] for region in room["regions"]]
        new_regions = [region for room in new_scene_graph['rooms'] for region in room["regions"]]
        matches, _ = detect_match(
            keys=old_regions,
            queries=new_regions,
            clip_model_list=self.clip_model_list,
            knn=3,
            overlap_relaxed=True,
            corr_score=None,
            unique_match=False,
        )

        no_caption_regions = []
        if len(matches) == 0:
            no_caption_regions = new_regions
        else:
            for match in matches:
                # check objects
                is_object_match = True
                query_object_ids = [obj['id'] for obj in match['q']['objects']]
                key_object_ids = [obj['id'] for obj in match['k']['objects']]
                for obj_i in match['k']['objects']:
                    if not obj_i['id'] in query_object_ids:
                        is_object_match = False
                        break
                for obj_j in match['q']['objects']:
                    if not obj_j['id'] in key_object_ids:
                        is_object_match = False
                        break
                if is_object_match:
                    match['q']['caption'] = match['k']['caption']
                    match['q']['corr_score'] = match['k']['corr_score']
            for region in new_regions:
                if region['caption']=='unknown':
                    no_caption_regions.append(region)
        if len(no_caption_regions)==0:
            return None
        return {'rooms': [{
            "regions": no_caption_regions,
        }]}

    def group_objects(self, objects, grid_size, origins_grid, gradient_map):
        wall_map = (gradient_map>1.2).astype(np.uint8)
        new_regions = detect_regions(objects, wall_map, knn_object=5, distance_threshold=50, wall_threshold=15, min_num=1, pred_sg=self)
        # n_obj_in_region = np.sum([len(x['objects']) for x in new_regions])
        # print(len(objects), n_obj_in_region)
        for region in new_regions:
            update_region(region, grid_size, origins_grid)
        scene_graph = {
            'rooms': [{
                "caption": 'apartment_room',
                "id": '0',
                "corr_score": 0.0,
                "center": origins_grid,
                "regions": new_regions,
                "objects": [],
            }]
        }
        return scene_graph, len(new_regions)

    def update_image_idx(self, region):
        objects = region['objects']
        image_idx1 = objects[0]["image_idx"]
        for obj2 in objects[1:]:
            image_idx2 = obj2["image_idx"]
            image_idx1 = set(image_idx1) | set(image_idx2)
        if len(image_idx1) == 0:
            return None
        conf_list = []
        for idx in image_idx1:
            conf_all = 0.0
            for obj in objects:
                if idx not in obj["image_idx"]:
                    continue
                conf = obj["conf"][obj["image_idx"].index(idx)]
                conf_all += conf
            conf_list.append(conf_all)
        region['image_idx'] = list(image_idx1)[np.argmax(np.array(conf_list))]

    def init_from_json(self, json_file):
        if isinstance(json_file, str):
            with open(json_file, 'r') as f:
                self.raw_scene_graph = json.load(f)
                self.scene_graph = self.raw_scene_graph
        elif isinstance(json_file, dict):
            self.raw_scene_graph = json_file
            self.scene_graph = json_file
        if "rooms" in self.scene_graph:
            self.scene_graph = self.assign_ids(self.scene_graph)
    
    def assign_ids(self, scene_graph):
        for room_id, room in enumerate(scene_graph["rooms"]):
            if "id" not in room:
                room["id"] = f'{room_id}'
            for region_id, region in enumerate(room["regions"]):
                if "id" not in region:
                    region["id"] = f'{room_id}.{region_id}'
                for obj_id, obj in enumerate(region["objects"]):
                    if "id" not in obj:
                        obj["id"] = f'{room_id}.{region_id}.{obj_id}'
        return scene_graph

    def init_obj_node(self, obj, target):
        if isinstance(obj, dict):
            # vln object
            points = np.asarray(obj['pcd'].points)
            center = points.mean(axis=0)
            x = int(center[0] * 100 / self.grid_size + self.origins_grid[0])
            y = int(center[2] * 100 / self.grid_size + self.origins_grid[1])
            object_node = {
                'caption': obj["caption"],
                'id': obj['global_id'],
                'center': [x, y],
                'confidence': float(max(obj['conf'])),
                'image_idx': obj['image_idx'],
                'conf': obj['conf'],
                'class_id': obj['class_id'],
            }
            if self.object_correlation == 'text':
                object_node["corr_score"] = self.get_text_sim_score(obj["caption"], target)
            elif self.object_correlation == 'visual':
                object_node["corr_score"] = self.get_vis_sim_score(obj['clip_ft'], target)
            elif self.object_correlation == 'llm':
                object_node["corr_score"] = self.get_llm_correlation(obj["caption"], target)
            elif self.object_correlation == 'cooccurrence':
                object_node["corr_score"] = self.get_cooccurrence_object(obj["caption"], target)
        else:
            # memory object node
            object_node = {
                'caption': obj.caption,
                'id': obj.id,
                'center': list(obj.center),
                'confidence': float(obj.object['conf'][-1]),
            }
            if hasattr(obj, "vis_sim_score"):
                object_node["corr_score"] = obj.vis_sim_score
            elif hasattr(obj, "text_sim_score"):
                object_node["corr_score"] = obj.text_sim_score
            else:
                obj.text_sim_score = self.get_text_sim_score(obj["caption"], target)
                object_node["corr_score"] = obj.text_sim_score
        obj['node'] = object_node
        return object_node

    def init_from_sg(self, scenegraph, target):
        room_nodes = []
        group_nodes = []
        nodes = []
        """Initializes the scene graph from a dictionary representation."""
        for room_id, room in enumerate(scenegraph.room_nodes):
            room_node = {
                "caption": room.caption,
                "id": f'{room_id}',
                "center": list(room.center) if room.center is not None else '',
                "regions": [],
                "objects": [],
            }
            room_node["corr_score"] = self.get_text_sim_score(room.caption, target)

            # Process objects in the room
            # for obj_id, obj in enumerate(room.nodes):
            #     object_node = self.init_obj_node(obj, target)
            #     nodes.append(object_node)
            #     room_node["objects"].append(object_node)

            # Process regions in the room
            for region_id, region in enumerate(room.group_nodes):
                group_node = {
                    "caption": region.caption,
                    "id": f'{room_id}.{region_id}',
                    "center": list(region.center) if region.center is not None else '',
                    "objects": [],
                    "image_idx": region.image_idx,
                }
                if not hasattr(region, "corr_score"):
                    # region.corr_score = self.get_llm_corr_score(region.caption, target)
                    # using clip text feature 
                    region.corr_score = self.get_text_sim_score(region.caption, target)
                group_node["corr_score"] = region.corr_score
                
                for obj_id, obj in enumerate(region.nodes):
                    object_node = self.init_obj_node(obj, target, f'{room_id}.{region_id}.{obj_id}')
                    group_node["objects"].append(object_node)
                    nodes.append(object_node)

                group_nodes.append(group_node)
                room_node["regions"].append(group_node)

            room_nodes.append(room_node)
        return room_nodes, group_nodes, nodes
    
    def merge_objects(self, scene_graph, distance_threshold=15.0, min_samples=1):
        """
        Merges nearby objects with the same caption into a single centered node and updates the scene graph.
        
        Args:
            scene_graph (dict): The input scene graph.
            distance_threshold (float): Maximum distance for clustering.
            min_samples (int): Minimum number of samples in a cluster.

        Returns:
            dict: The updated scene graph with merged objects.
        """
        for room in scene_graph["rooms"]:
            for region in room.get("regions", []):
                objects = region.get("objects", [])
                
                # Group objects by caption
                caption_groups = defaultdict(list)
                for obj in objects:
                    caption_groups[obj['caption']].append(obj)
                
                merged_objects = []
                for caption, group in caption_groups.items():
                    num_objects = len(group)
                    if num_objects == 1:
                        merged_objects.append(group[0])
                        continue
                    
                    positions = np.array([obj['center'] for obj in group])
                    clustering = DBSCAN(eps=distance_threshold, min_samples=min_samples, metric='euclidean').fit(positions)
                    labels = clustering.labels_
                    
                    cluster_centers = {}
                    cluster_coor_scores = {}
                    cluster_confidence_scores = {}
                    cluster_ids = {}
                    cluster_counts = defaultdict(int)
                    
                    for obj, label in zip(group, labels):
                        if label == -1:
                            merged_objects.append(obj)
                        else:
                            if label not in cluster_centers:
                                cluster_centers[label] = np.array(obj['center'])
                                cluster_ids[label] = obj['id']
                                cluster_coor_scores[label] = obj['corr_score']
                                cluster_confidence_scores[label] = obj['confidence']
                            else:
                                cluster_centers[label] += np.array(obj['center'])
                                cluster_coor_scores[label] += obj['corr_score']
                                cluster_confidence_scores[label] += obj['confidence']
                            cluster_counts[label] += 1
                    
                    for label, center in cluster_centers.items():
                        merged_objects.append({
                            "caption": caption,
                            "id": cluster_ids[label],
                            "center": (center / cluster_counts[label]).tolist(),
                            "corr_score": cluster_coor_scores[label] / cluster_counts[label],
                            "confidence": cluster_confidence_scores[label] / cluster_counts[label],
                        })
                
                region["objects"] = merged_objects
        
        return scene_graph

    def complete_scenegraph(self, scenegraph, target):
        """Initializes and validates the scene graph from a dictionary representation."""
        try:
            if not isinstance(scenegraph, dict):
                print(f'complete_scenegraph: scenegraph is not a dict')
                return None
            
            if "rooms" not in scenegraph or not isinstance(scenegraph["rooms"], list):
                print(f'complete_scenegraph: scenegraph has no rooms')
                return None
            
            for room in scenegraph["rooms"]:
                if not isinstance(room, dict):
                    print(f'complete_scenegraph: room is not a dict')
                    return None
                
                if "caption" not in room or not isinstance(room["caption"], str):
                    room["caption"] = ''
                
                if "center" not in room or not (isinstance(room["center"], list) and len(room["center"]) == 2):
                    room["center"] = [0, 0]
                
                if "corr_score" not in room or not isinstance(room.get("corr_score"), (int, float)):
                    room["corr_score"] = self.get_text_sim_score(room.get("caption", ''), target)
                
                if "regions" not in room or not isinstance(room["regions"], list):
                    room["regions"] = []
                
                for region in room["regions"]:
                    if not isinstance(region, dict):
                        print(f'complete_scenegraph: region is not a dict')
                        return None
                    
                    if "caption" not in region or not isinstance(region["caption"], (str, dict)):
                        region["caption"] = ''
                    
                    if "center" not in region or not (isinstance(region["center"], list) and len(region["center"]) == 2):
                        region["center"] = [0, 0]
                    
                    if "corr_score" not in region or not isinstance(region.get("corr_score"), (int, float)):
                        # region["corr_score"] = self.get_text_sim_score(region.get("caption", ''), target)
                        clip_model_list = (self.clip_model, self.clip_preprocess, self.clip_tokenizer)
                        if self.region_correlation == 'clip':
                            _, _, region["corr_score"] = get_text_sim_score(clip_model_list, region.get("caption", ''), target, topk=2)
                        elif self.region_correlation == 'cooccurrence':
                            if isinstance(region['caption'], dict):
                                region['corr_score'] = []
                                for caption, conf in region['caption'].items():
                                    region['corr_score'].append(self.get_cooccurrence_region(caption, self.target))
                            else:
                                region['corr_score'] = self.get_cooccurrence_region(region['caption'], self.target)
                    
                    if "objects" not in region or not isinstance(region["objects"], list):
                        region["objects"] = []
                    
                    for obj in region["objects"]:
                        if not isinstance(obj, dict):
                            print(f'complete_scenegraph: obj is not a dict')
                            return None
                        
                        if "caption" not in obj or not isinstance(obj["caption"], str):
                            obj["caption"] = ''
                        
                        if "center" not in obj or not (isinstance(obj["center"], list) and len(obj["center"]) == 2):
                            obj["center"] = [0, 0]
                        
                        if "corr_score" not in obj or not isinstance(obj.get("corr_score"), (int, float)):
                            obj["corr_score"] = self.get_text_sim_score(obj.get("caption", ''), target)

                        if "confidence" not in obj:
                            try:
                                if obj.get("confidence") is None:
                                    pass
                                elif isinstance(obj.get("confidence"), str):
                                    float(obj.get("confidence"))
                                elif not isinstance(obj.get("confidence"), (int, float)):
                                    pass
                            except Exception:
                                pass
            
            return scenegraph
        except Exception as e:
            print(f'complete_scenegraph: error in completing scenegraph: {e}')
            return None

    def get_cooccurrence_object(self, obj_label, target_label):
        if obj_label == target_label:
            return 1
        elif obj_label in categories_21_plus_stairs and target_label in categories_21_plus_stairs:
            return self.heatmap_obj_obj[categories_21_plus_stairs.index(obj_label), categories_21_plus_stairs.index(target_label)]
        else:
            print(f"Error: {obj_label} or {target_label} is not in the table (get_cooccurrence_object)")
            return 0

    def get_cooccurrence_region(self, region_label, target_label):
        region_label = region_label.replace('_', ' ')
        if region_label.lower().strip() == 'unknown':
            return 0
        elif region_label in region_captions and target_label in categories_21_plus_stairs:
            return self.heatmap_region_obj[region_captions.index(region_label), categories_21_plus_stairs.index(target_label)]
        else:
            print(f"Error: {region_label} or {target_label} is not in the table (get_cooccurrence_region)")
            return 0
    
    def get_llm_correlation(self, obj_label, target_label):
        if obj_label == target_label:
            return 1
        elif obj_label in categories_21 and target_label in categories_21:
            return self.corr_matrix_obj_llm[categories_21.index(obj_label), categories_21.index(target_label)]
        else:
            print(f"Error: {obj_label} or {target_label} is not in the table (get_llm_correlation)")
            return 0

    def get_vis_sim_score(self, vis_feat, target):
        target_feat = self.get_text_feat(target)
        vis_sim_score = vis_feat @ target_feat.T
        return vis_sim_score.item()
    
    def get_text_sim_score(self, obs, target):
        obs_feat = self.get_text_feat(obs)
        target_feat = self.get_text_feat(target)
        sim_score = obs_feat @ target_feat.T
        return sim_score.item()

    def get_vis_feat(self, image, image_idx=None):
        if image_idx is not None and image_idx in self.clip_visual_cache:
            return self.clip_visual_cache[image_idx]
        # If not cached, compute the visual features
        with torch.no_grad():
            image_batch = self.clip_preprocess(Image.fromarray(image)).to("cuda").unsqueeze(0).half()
            image_feats = self.clip_model.encode_image(image_batch)
            image_feats /= image_feats.norm(dim=-1, keepdim=True)
            if image_idx is not None:
                self.clip_visual_cache[image_idx] = image_feats
        return image_feats

    def get_text_feat(self, text):
        if text in self.clip_textual_cache:
            return self.clip_textual_cache[text]
        with torch.no_grad():
            # TODO: debug text too long, it gives the all thought process instead of a caption
            tokenized_text = self.clip_tokenizer([text[:77]]).to("cuda")
            text_feat = self.clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            self.clip_textual_cache[text] = text_feat
        return text_feat
    
    def get_llm_corr_score(self, observation, target):
        llm_prompt = construct_query(self.prompts["RegionCorrEstimatior"], observation=observation, target=target)
        response = query_llm(llm_prompt, llm_name='llama3.2-vision')
        corr_score = text2value(response)
        return corr_score

    def prune_subgraph(self, scene_graph, frontier_location, threshold=20, pred_threshold=40):
        """
        Extracts only relevant objects near the frontier and their parent nodes
        using a bottom-up approach.
        
        Bottom-up steps:
        1. Filter objects in each region and in the room based on distance.
        2. For each region, include it if its center is near the frontier OR it
            contains any relevant objects.
        3. For each room, include it if its center is near the frontier OR if it
            contains any relevant objects or regions.
        """
        subgraph = {"rooms": []}
        if scene_graph is None:
            return subgraph
        
        pred_region_count = 0
        best_pred_region_dist = float('inf')
        best_pred_region = None

        for room in scene_graph.get("rooms", []):
            # --- Process room-level objects ---
            filtered_room_objects = []
            for obj in room.get("objects", []):
                # Use object's center; if missing, fallback to room center (or [0,0])
                obj_center = np.array(obj.get("center", room.get("center", [0, 0])))
                obj_distance = np.linalg.norm(obj_center - frontier_location)
                if obj_distance < threshold:
                    filtered_room_objects.append(obj)

            # --- Process regions (child nodes) ---
            filtered_regions = []
            for region in room.get("regions", []):

                # Process objects within the region
                filtered_region_objects = []
                for obj in region.get("objects", []):
                    # Use object's center; if missing, fallback to region center then room center
                    obj_center = np.array(obj.get("center", region.get("center", room.get("center", [0, 0]))))
                    obj_distance = np.linalg.norm(obj_center - frontier_location)
                    if obj_distance < threshold:
                        filtered_region_objects.append(obj)
                
                # Determine region's own relevance by its center
                if region.get("center", None) is not None:
                    region_center = np.array(region.get("center", room.get("center", [0, 0])))
                    region_distance = np.linalg.norm(region_center - frontier_location)
                else:
                    region_distance = float('inf')

                # If the region is predicted, include it
                if region.get("predicted", False) and region_distance < threshold:
                    filtered_regions.append(region)
                    pred_region_count += 1
                    continue
                if region_distance < best_pred_region_dist:
                    best_pred_region_dist = region_distance
                    best_pred_region = region
                
                # Include region if its center is close OR if it contains any relevant objects
                if region_distance < threshold or filtered_region_objects:
                    filtered_region = deepcopy(region)
                    filtered_region["objects"] = filtered_region_objects
                    filtered_regions.append(filtered_region)

            # --- Add the closest predicted region ---
            if pred_region_count == 0 and best_pred_region_dist < pred_threshold:
                filtered_regions.append(best_pred_region)
            
            # --- Decide whether to include the room ---
            if room.get("center", None) is not None:
                room_center = np.array(room.get("center", [0, 0]))
                room_distance = np.linalg.norm(room_center - frontier_location)
            else:
                room_distance = float('inf')
            
            # Include the room if:
            #  - Its center is within threshold, OR
            #  - It has any relevant room-level objects, OR
            #  - It has any relevant regions (with relevant objects)
            if room_distance < threshold or filtered_room_objects or filtered_regions:
                filtered_room = deepcopy(room)
                filtered_room["objects"] = filtered_room_objects
                filtered_room["regions"] = filtered_regions
                subgraph["rooms"].append(filtered_room)

        return subgraph

    def merge_subgraph(self, observed_graph, predicted_graph):
        """
        Merges the predicted scene graph with the observed one, ensuring consistency.
        This function handles merging at the level of rooms, regions, and objects,
        and avoids duplicate entries.
        """

        # Make a deep copy of the observed graph to avoid modifying it directly
        merged_graph = deepcopy(observed_graph)

        # Build a mapping for observed rooms by room id.
        try:
            room_map = {room["id"]: room for room in merged_graph.get("rooms", [])}
        except:
            print(f"Error: {merged_graph}")
            return None

        for new_room in predicted_graph.get("rooms", []):
            room_id = new_room.get("id")
            if room_id in room_map:
                existing_room = room_map[room_id]
                # Merge room-level objects.
                existing_obj_ids = {obj.get("id") for obj in existing_room.get("objects", [])}
                for new_obj in new_room.get("objects", []):
                    new_obj_id = new_obj.get("id")
                    if new_obj_id not in existing_obj_ids:
                        existing_room.setdefault("objects", []).append(new_obj)
                        existing_obj_ids.add(new_obj_id)

                # Merge regions within the room.
                # Use a key that combines region id and caption (if available) to identify uniqueness.
                region_map = {}
                for region in existing_room.get("regions", []):
                    caption = region.get("caption", "")
                    if isinstance(caption, dict):
                        caption_str = str(sorted(caption.items()))  # Ensure deterministic ordering
                    else:
                        caption_str = str(caption)
                    region_map[(region.get("id"), caption_str)] = region

                for new_region in new_room.get("regions", []):
                    new_caption = new_region.get("caption", "")
                    if isinstance(new_caption, dict):
                        new_caption_str = str(sorted(new_caption.items()))
                    else:
                        new_caption_str = str(new_caption)
                    region_key = (new_region.get("id"), new_caption_str)

                    if region_key in region_map:
                        existing_region = region_map[region_key]
                        # Merge objects within the region.
                        existing_region_obj_ids = {obj.get("id") for obj in existing_region.get("objects", [])}
                        for new_region_obj in new_region.get("objects", []):
                            if isinstance(new_region_obj, dict):
                                new_region_obj_id = new_region_obj.get("id")
                            elif isinstance(new_region_obj, str):
                                new_region_obj = self.init_obj_node(new_region_obj, self.target,
                                                                    obj_id=f'{room_id}.{new_region.get("id")}.new')
                                new_region_obj_id = new_region_obj.get("id")
                            else:
                                continue  # Skip if it's not an expected type.

                            if new_region_obj_id not in existing_region_obj_ids:
                                existing_region.setdefault("objects", []).append(new_region_obj)
                                existing_region_obj_ids.add(new_region_obj_id)
                    else:
                        # Region does not exist in the observed room: add the entire region.
                        new_region["predicted"] = True
                        existing_room.setdefault("regions", []).append(new_region)
            else:
                # Room does not exist in the observed graph: add it entirely.
                merged_graph.setdefault("rooms", []).append(new_room)

        return merged_graph


    def diff_scenegraph(self, previous, current):
        """
        Compare the previous and current scene graph and return the differences.
        Differences are reported at the room, region, and object levels.
        """
        def compare_rooms(prev_room, current_room):
            """Compare two rooms and return the differences in attributes."""
            room_diff = {}
            if prev_room.get("caption") != current_room.get("caption"):
                room_diff["caption"] = {"old": prev_room.get("caption"), "new": current_room.get("caption")}
            if prev_room.get("center") != current_room.get("center"):
                room_diff["center"] = {"old": prev_room.get("center"), "new": current_room.get("center")}
            if prev_room.get("connected_rooms") != current_room.get("connected_rooms"):
                room_diff["connected_rooms"] = {"old": prev_room.get("connected_rooms"), "new": current_room.get("connected_rooms")}
            return room_diff

        def compare_regions(prev_region, current_region):
            """Compare two regions and return the differences in attributes."""
            region_diff = {}
            if prev_region.get("caption") != current_region.get("caption"):
                region_diff["caption"] = {"old": prev_region.get("caption"), "new": current_region.get("caption")}
            if prev_region.get("center") != current_region.get("center"):
                region_diff["center"] = {"old": prev_region.get("center"), "new": current_region.get("center")}
            return region_diff

        def compare_objects(prev_obj, current_obj):
            """Compare two objects and return the differences in attributes."""
            obj_diff = {}
            if prev_obj.get("caption") != current_obj.get("caption"):
                obj_diff["caption"] = {"old": prev_obj.get("caption"), "new": current_obj.get("caption")}
            if prev_obj.get("center") != current_obj.get("center"):
                obj_diff["center"] = {"old": prev_obj.get("center"), "new": current_obj.get("center")}
            if prev_obj.get("type") != current_obj.get("type"):
                obj_diff["type"] = {"old": prev_obj.get("type"), "new": current_obj.get("type")}
            return obj_diff

        # Initialize diff result structure.
        diff = {
            "new_rooms": [],
            "removed_rooms": [],
            "updated_rooms": [],
            "new_regions": [],
            "removed_regions": [],
            "updated_regions": [],
            "new_objects": [],
            "removed_objects": [],
            "updated_objects": []
        }

        # Map rooms by their id for quick lookup.
        previous_rooms = {room.get("id", ''): room for room in previous.get("rooms", [])}
        current_rooms  = {room.get("id", ''): room for room in current.get("rooms", [])}

        # Process rooms.
        for room_id, curr_room in current_rooms.items():
            if room_id not in previous_rooms:
                diff["new_rooms"].append(curr_room)
            else:
                prev_room = previous_rooms[room_id]
                room_changes = compare_rooms(prev_room, curr_room)
                if room_changes:
                    diff["updated_rooms"].append({"room_id": room_id, "changes": room_changes})
                
                # Compare room-level objects.
                prev_room_objs = {obj["id"]: obj for obj in prev_room.get("objects", [])}
                curr_room_objs = {obj["id"]: obj for obj in curr_room.get("objects", [])}
                
                for obj_id, curr_obj in curr_room_objs.items():
                    if obj_id not in prev_room_objs:
                        diff["new_objects"].append({"room_id": room_id, "object": curr_obj})
                    else:
                        changes = compare_objects(prev_room_objs[obj_id], curr_obj)
                        if changes:
                            diff["updated_objects"].append({"room_id": room_id, "object_id": obj_id, "changes": changes})
                for obj_id, prev_obj in prev_room_objs.items():
                    if obj_id not in curr_room_objs:
                        diff["removed_objects"].append({"room_id": room_id, "object": prev_obj})
                
                # Compare regions.
                prev_regions = {region.get("id", ''): region for region in prev_room.get("regions", [])}
                curr_regions = {region.get("id", ''): region for region in curr_room.get("regions", [])}
                
                for region_id, curr_region in curr_regions.items():
                    if region_id not in prev_regions:
                        diff["new_regions"].append({"room_id": room_id, "region": curr_region})
                    else:
                        prev_region = prev_regions[region_id]
                        region_changes = compare_regions(prev_region, curr_region)
                        if region_changes:
                            diff["updated_regions"].append({"room_id": room_id, "region_id": region_id, "changes": region_changes})
                        
                        # Compare objects inside regions.
                        prev_reg_objs = {obj.get("id", ''): obj for obj in prev_region.get("objects", [])}
                        curr_reg_objs = {obj.get("id", ''): obj for obj in curr_region.get("objects", [])}
                        
                        for obj_id, curr_obj in curr_reg_objs.items():
                            if obj_id not in prev_reg_objs:
                                diff["new_objects"].append({"room_id": room_id, "region_id": region_id, "object": curr_obj})
                            else:
                                changes = compare_objects(prev_reg_objs[obj_id], curr_obj)
                                if changes:
                                    diff["updated_objects"].append({"room_id": room_id, "region_id": region_id, "object_id": obj_id, "changes": changes})
                        for obj_id, prev_obj in prev_reg_objs.items():
                            if obj_id not in curr_reg_objs:
                                diff["removed_objects"].append({"room_id": room_id, "region_id": region_id, "object": prev_obj})
                                
                for region_id, prev_region in prev_regions.items():
                    if region_id not in curr_regions:
                        diff["removed_regions"].append({"room_id": room_id, "region": prev_region})
        
        # Identify removed rooms.
        for room_id, prev_room in previous_rooms.items():
            if room_id not in current_rooms:
                diff["removed_rooms"].append(prev_room)

        return diff

    def predict_scenegraph(self, sub_graph, frontier_location, llm_name = "qwen2.5"):
        vln_logger.increase_function_call_count('predict_scenegraph', args={'mode': 'detailed'})
        if len(sub_graph.get("rooms", [])) == 0:
            return sub_graph
        scene_description = json.dumps(sub_graph, indent=4)
        # vln_logger.info(f"Sub scene graph: {sub_graph}", extra={"module": "SceneGraphPrediction"})

        # 2. Query LLM to predict the observations and covert the text response to the graph
        llm_prompt = construct_query(self.prompts["SceneGraphComplementor"], target=self.target, frontier_location=frontier_location, scene_description=scene_description)
        response = query_llm(llm_prompt, llm_name=llm_name) # qwen2.5:32b, deepseek-r1:14b, phi4
        # print('LLM response: ', response)
        predicted_graph = format_graph(response)
        
        if predicted_graph is None:
            return sub_graph
        # print('Pred graph: ', predicted_graph)

        try:
            completed_graph = self.complete_scenegraph(predicted_graph, self.target)
            # print('Completed graph: ', completed_graph)
        except:
            completed_graph = sub_graph
        return completed_graph
    
    def predict_highlevel_scenegraph(self, sub_graph, frontier_location, llm_name = "qwen2.5"):
        '''
        Only predict the high-level 
        '''
        vln_logger.increase_function_call_count('predict_scenegraph', args={'mode': 'highlevel'})
        if len(sub_graph.get("rooms", [])) == 0:
            return sub_graph
        
        scene_description = self.get_region_description(sub_graph)
        # vln_logger.info(f"Sub scene graph: {sub_graph}", extra={"module": "SceneGraphPrediction"})
        # 2. Query LLM to predict the observations and covert the text response to the graph
        llm_prompt = construct_query(self.prompts["HighLevelSceneGraphComplementor"], target=self.target, scene_description=scene_description, frontier_location=frontier_location)
        vln_logger.info("Querying LLM for high-level scene graph", extra={"module": "SceneGraphPrediction"})
        response = query_llm(llm_prompt, llm_name=llm_name) # qwen2.5:32b, deepseek-r1:14b, phi4
        # print('LLM response: ', response)
        predicted_regions = format_graph(response)
        if predicted_regions is None:
            return None
        # print('LLM response: ', predicted_regions)
        # vln_logger.info(f"Predicted scene graph: {predicted_regions}", extra={"module": "SceneGraphPrediction"})
        center = np.array([region.get("center", [0,0]) for region in predicted_regions.get("regions", [])]).reshape(-1, 2).mean(axis=0)
        caption = ''.join(list(set([region.get("caption", '') for region in predicted_regions.get("regions", [])])))
        predicted_regions.update({
            'caption': caption,
            'center': list(center),
        })
        predicted_graph = {"rooms": [predicted_regions]}

        try:
            completed_graph = self.complete_scenegraph(predicted_graph, self.target)
            # print('Completed graph: ', completed_graph)
        except:
            completed_graph = sub_graph
        sub_scene_graph = self.prune_subgraph(copy.deepcopy(completed_graph), frontier_location, threshold=32)
        return sub_scene_graph


    def get_region_description(self, sub_graph):
        region_description = ''
        regions = []
        for room in sub_graph["rooms"]:
            regions.extend(room.get("regions", []))
        if len(regions) == 0:
            return 'None'
        for i, region in enumerate(regions):
            objects = [obj["caption"] for obj in region['objects']]
            region_description += f"Region {region['id']}: {region['caption']} center: {region['center']} contained objects: {objects} \n"
        return region_description
    
    def get_frontier_description(self, frontier_locations):
        frontier_description = ''
        for i, location in enumerate(frontier_locations):
            frontier_description += f"Unknown region {i} center: {location} \n"

    def get_frontier_with_regions_description(self, frontier_locations, observed_graph):
        frontier_description = ''
        obs_regions = [region for room in observed_graph['rooms'] for region in room['regions'] if not region.get('predicted', False)]
        nearby_regions = get_nearby_regions(frontier_locations, obs_regions, knn=3, distance=80)
        for i, (location, nearby_regions) in enumerate(zip(frontier_locations, nearby_regions)):
            nearby_regions_description = self.get_region_description({'rooms': [{'regions': nearby_regions}]})
            frontier_description += f"Unknown region {i}, center: {location}, nearby regions:\n{nearby_regions_description} \n"+'-'*10+'\n'
        return frontier_description
    
    

    def predict_global_scenegraph(self, dump_dir, occupancy_map, observed_graph, frontier_locations, agent_location, llm_name = "qwen2.5", vlm=False):
        '''
            Scene graph prediction for all the frontiers
        '''
        H, W = occupancy_map.shape[:2]
        max_try_times = 3
        def _coord_transform(coords):
            if isinstance(coords, dict):    
                for room in coords["rooms"]:
                    if room.get("center", None) is not None:
                        room["center"] = [room["center"][1], H-room["center"][0]]
                    for region in room.get("regions", []):
                        if region.get("center", None) is not None:
                            region["center"] = [region["center"][1], H-region["center"][0]]
                        for obj in region.get("objects", []):
                            if obj.get("center", None) is not None:
                                obj["center"] = [obj["center"][1], H-obj["center"][0]]
                return coords
            elif isinstance(coords, list) or isinstance(coords, np.ndarray):
                frontier_locations_y = H - coords[:, 0]
                frontier_locations_x = coords[:, 1]
                frontier_locations = np.stack((frontier_locations_x, frontier_locations_y), axis=1)
                return frontier_locations
        
        def _reverse_coord_transform(scene_graph):
            for room in scene_graph["rooms"]:
                if room.get("center", None) is not None:
                    room["center"] = [H-room["center"][1], room["center"][0]]
                for region in room.get("regions", []):
                    if region.get("center", None) is not None:
                        region["center"] = [H-region["center"][1], region["center"][0]]
                    for obj in region.get("objects", []):
                        if obj.get("center", None) is not None:
                            obj["center"] = [H-obj["center"][1], obj["center"][0]]
            return scene_graph
        
        vln_logger.increase_function_call_count('predict_scenegraph', args={'mode': 'global'})
        if observed_graph is None or len(observed_graph.get("rooms", [])) == 0:
            return observed_graph
        observed_regions = []
        for room in observed_graph["rooms"]:
            observed_regions.extend(room.get("regions", []))
        if len(observed_regions) == 0:
            return observed_graph
        
        
        # vln_logger.info(f"Sub scene graph: {sub_graph}", extra={"module": "SceneGraphPrediction"})
        # 2. Query LLM to predict the observations and covert the text response to the graph
        if vlm:
            trans_scene_graph = _coord_transform(copy.deepcopy(observed_graph))
            scene_description = self.get_region_description(trans_scene_graph)
            save_path = f"{dump_dir}bev_{llm_name}/{vln_logger.scene}_{vln_logger.episode}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            bev_image_path = f"{save_path}/{vln_logger.step}_occ_map.png"
            if not os.path.exists(bev_image_path):
                try:
                    bev_image = visualize_BEV(occupancy_map, observed_graph, bev_image_path, target_locations=frontier_locations, agent_location=agent_location)
                except:
                    print('frontier_locations: ', frontier_locations)
                vln_logger.update_bev_scenegraph(bev_image)
                if vln_logger.sg_pred_image is None:
                    vln_logger.update_pred_bev_scenegraph(bev_image)
            transformed_frontier_locations = _coord_transform(deepcopy(frontier_locations))
            transformed_agent_location = [agent_location[1], H-agent_location[0]]
            llm_prompt = construct_query(self.prompts["GlobalBEVRGBSceneGraphComplementor"], target=self.target, scene_description=scene_description, agent_location=transformed_agent_location, frontier_locations=transformed_frontier_locations)
            for _ in range(max_try_times):
                response = query_llm(llm_prompt, llm_name=llm_name, image=bev_image_path) # qwen2.5:32b, deepseek-r1:14b, phi4
                predicted_regions = format_graph(response)
                if predicted_regions is not None:
                    break
        else:
            scene_description = self.get_region_description(observed_graph)
            llm_prompt = construct_query(self.prompts["GlobalSceneGraphComplementor"], target=self.target, scene_description=scene_description, frontier_locations=frontier_locations)
            for _ in range(max_try_times):
                response = query_llm(llm_prompt, llm_name=llm_name) # qwen2.5:32b, deepseek-r1:14b, phi4
                predicted_regions = format_graph(response)
                if predicted_regions is not None:
                    break
        if predicted_regions is None:
            return None
        # print('LLM response: ', predicted_regions)
        # vln_logger.info(f"Predicted scene graph: {predicted_regions}", extra={"module": "SceneGraphPrediction"})
        try:
            center = np.array([region.get("center", [0,0]) for region in predicted_regions.get("regions", [])]).reshape(-1, 2).mean(axis=0)
            caption = ''.join(list(set([region.get("caption", '') for region in predicted_regions.get("regions", [])])))
            predicted_regions.update({
                'caption': caption,
                'center': list(center),
            })
            predicted_graph = {"rooms": [predicted_regions]}
        except:
            return None

        completed_graph = self.complete_scenegraph(predicted_graph, self.target)
        if completed_graph is None:
            return None
        if vlm:
            try:
                completed_graph = _reverse_coord_transform(copy.deepcopy(completed_graph))
            except:
                print("Error in reversing coordinate transformation: " , completed_graph)
                return None
        # vln_logger.info(f"Predicted scene graph: {completed_graph}", extra={"module": "SceneGraphPrediction"})

        # 3. Merge the predicted subgraph with original scene graph
        self.predicted_sg = completed_graph
        merged_graph = self.merge_subgraph(observed_graph, completed_graph)
        vln_logger.info(f"Merged scene graph: {merged_graph}", extra={"module": "SceneGraphPrediction"})
        
        if vlm:
            pred_bev_image_path = f"{save_path}/{vln_logger.step}_occ_map_pred.png"
            if merged_graph is not None and not os.path.exists(pred_bev_image_path):
                bev_image = visualize_BEV(occupancy_map, merged_graph, pred_bev_image_path, target_locations=frontier_locations, agent_location=agent_location)
                vln_logger.update_pred_bev_scenegraph(bev_image)
        return merged_graph

    def predict_global_scenegraph_TopK(self, dump_dir, occupancy_map, observed_graph, frontier_locations, agent_location, llm_name = "qwen2.5", vlm=False):
        '''
            Scene graph prediction for all the frontiers
        '''
        H, W = occupancy_map.shape[:2]
        max_try_times = 3
        def _coord_transform(coords):
            if isinstance(coords, dict):    
                for room in coords["rooms"]:
                    if room.get("center", None) is not None:
                        room["center"] = [room["center"][1], H-room["center"][0]]
                    for region in room.get("regions", []):
                        if region.get("center") is not None and len(region.get("center")) > 0:
                            region["center"] = [region["center"][1], H-region["center"][0]]
                        for obj in region.get("objects", []):
                            if obj.get("center", None) is not None:
                                obj["center"] = [obj["center"][1], H-obj["center"][0]]
                return coords
            elif isinstance(coords, list) or isinstance(coords, np.ndarray):
                frontier_locations_y = H - coords[:, 0]
                frontier_locations_x = coords[:, 1]
                frontier_locations = np.stack((frontier_locations_x, frontier_locations_y), axis=1)
                return frontier_locations
        
        def _reverse_coord_transform(scene_graph):
            for room in scene_graph["rooms"]:
                if room.get("center", None) is not None:
                    room["center"] = [H-room["center"][1], room["center"][0]]
                for region in room.get("regions", []):
                    if region.get("center", None) is not None:
                        region["center"] = [H-region["center"][1], region["center"][0]]
                    for obj in region.get("objects", []):
                        if obj.get("center", None) is not None:
                            obj["center"] = [H-obj["center"][1], obj["center"][0]]
            return scene_graph
        
        vln_logger.increase_function_call_count('predict_scenegraph', args={'mode': 'global'})
        if observed_graph is None or len(observed_graph.get("rooms", [])) == 0:
            return observed_graph
        
        print(f"before merge: {len(observed_graph['rooms'][0]['regions'])}")
        for room in observed_graph["rooms"]:
            room["regions"] = merge_regions(room["regions"], self.clip_model_list, self.grid_size, self.origins_grid)
        print(f"after merge: {len(observed_graph['rooms'][0]['regions'])}")
        
        # vln_logger.info(f"Sub scene graph: {sub_graph}", extra={"module": "SceneGraphPrediction"})
        # 2. Query LLM to predict the observations and covert the text response to the graph
        if vlm:
            trans_scene_graph = _coord_transform(copy.deepcopy(observed_graph))
            scene_description = self.get_region_description(trans_scene_graph)
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # bev_image_path = f"{save_path}/{vln_logger.step}_occ_map.png"
            # if not os.path.exists(bev_image_path):
            self.obs_bev_image = visualize_BEV(occupancy_map, observed_graph, None, target_locations=frontier_locations, agent_location=agent_location)
            # vln_logger.update_bev_scenegraph(bev_image)
            # if vln_logger.sg_pred_image is None:
            #     vln_logger.update_pred_bev_scenegraph(bev_image)
            transformed_frontier_locations = _coord_transform(deepcopy(frontier_locations))
            transformed_agent_location = [agent_location[1], H-agent_location[0]]
            llm_prompt = construct_query(self.prompts["GlobalBEVRGBSceneGraphComplementor_TopK"], target=self.target, scene_description=scene_description, agent_location=transformed_agent_location, frontier_locations=transformed_frontier_locations)
            self._llm_prompt = llm_prompt
            for _ in range(max_try_times):
                response = query_llm(llm_prompt, llm_name=llm_name, image=self.obs_bev_image) # qwen2.5:32b, deepseek-r1:14b, phi4
                predicted_regions = format_graph(response)
                if predicted_regions is not None:
                    self._response = response
                    self._predicted_regions = predicted_regions
                    break
        else:
            scene_description = self.get_region_description(observed_graph)
            llm_prompt = construct_query(self.prompts["GlobalSceneGraphComplementor"], target=self.target, scene_description=scene_description, frontier_locations=frontier_locations)
            self._llm_prompt = llm_prompt
            for _ in range(max_try_times):
                response = query_llm(llm_prompt, llm_name=llm_name) # qwen2.5:32b, deepseek-r1:14b, phi4
                predicted_regions = format_graph(response)
                if predicted_regions is not None:
                    self._response = response
                    self._predicted_regions = predicted_regions
                    break
        if predicted_regions is None:
            return None
        # print('LLM response: ', predicted_regions)
        # vln_logger.info(f"Predicted scene graph: {predicted_regions}", extra={"module": "SceneGraphPrediction"})
        try:
            center = np.array([region.get("center", [0,0]) for region in predicted_regions.get("regions", [])]).reshape(-1, 2).mean(axis=0)
            # caption = ''.join(list(set([region.get("caption", '') for region in predicted_regions.get("regions", [])])))
            predicted_regions.update({
                'caption': 'room',
                'center': list(center),
            })
            predicted_graph = {"rooms": [predicted_regions]}
        except:
            return None
        self.predicted_regions = predicted_regions["regions"]

        completed_graph = self.complete_scenegraph(predicted_graph, self.target)
        if completed_graph is None:
            return None
        completed_graph = self.assign_ids(completed_graph)
        if vlm:
            try:
                completed_graph = _reverse_coord_transform(copy.deepcopy(completed_graph))
            except:
                print("Error in reversing coordinate transformation: " , completed_graph)
                return None
        # vln_logger.info(f"Predicted scene graph: {completed_graph}", extra={"module": "SceneGraphPrediction"})

        # 3. Merge the predicted subgraph with original scene graph
        self.predicted_sg = completed_graph
        merged_graph = self.merge_subgraph(observed_graph, completed_graph)
        vln_logger.info(f"Merged scene graph: {merged_graph}", extra={"module": "SceneGraphPrediction"})
        
        if vlm and merged_graph is not None:
            self.pred_bev_image = visualize_BEV(occupancy_map, merged_graph, None, target_locations=frontier_locations, agent_location=agent_location)
        return merged_graph
    
    def get_llm_results(self):
        if not hasattr(self, '_llm_prompt') or not hasattr(self, '_response') or not hasattr(self, '_predicted_regions'):
            return None
        return {
            'llm_prompt': self._llm_prompt,
            'response': self._response,
            'predicted_regions': self._predicted_regions,
        }

    def get_reasoning(self):
        if self.predicted_sg == {}:
            return []
        reasoning = [f'{region["id"]}: {region["reasoning"]}' for room in self.predicted_sg['rooms'] for region in room['regions']]
        return reasoning
    
    def predict_global_scenegraph_TopK_WithNumber(self, dump_dir, occupancy_map, observed_graph, frontier_locations, agent_location, llm_name = "qwen2.5", vlm=False):
        '''
            Scene graph prediction for all the frontiers
        '''
        print(f'[prediction] frontier_locations: {len(frontier_locations)}')
        if len(frontier_locations) == 0 or self.step < 30:
            return observed_graph
        H, W = occupancy_map.shape[:2]
        max_try_times = 3
        def _coord_transform(coords):
            if isinstance(coords, dict):    
                for room in coords["rooms"]:
                    if room.get("center", '') != '':
                        room["center"] = [room["center"][1], H-room["center"][0]]
                    for region in room.get("regions", []):
                        if region.get("center") is not None and len(region.get("center")) > 0:
                            region["center"] = [region["center"][1], H-region["center"][0]]
                        for obj in region.get("objects", []):
                            if obj.get("center", '') != '':
                                obj["center"] = [obj["center"][1], H-obj["center"][0]]
                return coords
            elif isinstance(coords, list) or isinstance(coords, np.ndarray):
                frontier_locations_y = H - coords[:, 0]
                frontier_locations_x = coords[:, 1]
                frontier_locations = np.stack((frontier_locations_x, frontier_locations_y), axis=1)
                return frontier_locations
        
        def _reverse_coord_transform(scene_graph):
            for room in scene_graph["rooms"]:
                if room.get("center", '') != '':
                    room["center"] = [H-room["center"][1], room["center"][0]]
                for region in room.get("regions", []):
                    if region.get("center", '') != '':
                        region["center"] = [H-region["center"][1], region["center"][0]]
                    for obj in region.get("objects", []):
                        if obj.get("center", '') != '':
                            obj["center"] = [H-obj["center"][1], obj["center"][0]]
            return scene_graph
        
        vln_logger.increase_function_call_count('predict_scenegraph', args={'mode': 'global'})
        if observed_graph is None or len(observed_graph.get("rooms", [])) == 0:
            print(f'No observed graph')
            return observed_graph
        
        # vln_logger.info(f"Sub scene graph: {sub_graph}", extra={"module": "SceneGraphPrediction"})
        # 2. Query LLM to predict the observations and covert the text response to the graph
        if vlm:
            _coord_transform(copy.deepcopy(observed_graph))
            bev_image = visualize_BEV(occupancy_map, observed_graph, None, target_locations=frontier_locations, agent_location=agent_location)
            bev_image = remove_image_border(bev_image[..., :3])
            self.obs_bev_image = bev_image
            transformed_frontier_locations = _coord_transform(deepcopy(frontier_locations))
            transformed_agent_location = [agent_location[1], H-agent_location[0]]
            frontier_description = self.get_frontier_with_regions_description(transformed_frontier_locations, observed_graph)
            # region_choices = [region_captions[i] for i, score in enumerate(region_corr_score) if score > 0.25]
            region_choices = chance_regions
            region_choices += ['unknown']
            region_choices_str = ', '.join(region_choices)
            llm_prompt = construct_query(self.prompts["GlobalBEVRGBSceneGraphComplementor_TopK_Number_WithCorrelation"], target=self.target, agent_location=transformed_agent_location, frontier_locations=frontier_description, region_choices=region_choices_str)
            self._llm_prompt = llm_prompt
            for _ in range(max_try_times):
                response = query_llm(llm_prompt, llm_name=llm_name, image=bev_image) # qwen2.5:32b, deepseek-r1:14b, phi4
                predicted_regions = format_graph(response)
                if predicted_regions is not None:
                    self._response = response
                    self._predicted_regions = predicted_regions
                    break
        else:
            scene_description = self.get_region_description(observed_graph)
            llm_prompt = construct_query(self.prompts["GlobalSceneGraphComplementor"], target=self.target, scene_description=scene_description, frontier_locations=frontier_locations)
            self._llm_prompt = llm_prompt
            for _ in range(max_try_times):
                response = query_llm(llm_prompt, llm_name=llm_name) # qwen2.5:32b, deepseek-r1:14b, phi4
                predicted_regions = format_graph(response)
                if predicted_regions is not None:
                    self._response = response
                    self._predicted_regions = predicted_regions
                    break
        if predicted_regions is None:
            print(f'No predicted regions')
            return None
        # print('LLM response: ', predicted_regions)
        # vln_logger.info(f"Predicted scene graph: {predicted_regions}", extra={"module": "SceneGraphPrediction"})
        try:
            center = np.array([region.get("center", [0,0]) for region in predicted_regions.get("regions", [])]).reshape(-1, 2).mean(axis=0)
            # caption = ''.join(list(set([region.get("caption", '') for region in predicted_regions.get("regions", [])])))
            predicted_regions.update({
                'caption': 'room',
                'center': list(center),
            })
            predicted_graph = {"rooms": [predicted_regions]}
        except:
            print(f'Error in parsing predicted regions')
            return None

        self.predicted_regions = predicted_regions["regions"]

        completed_graph = self.complete_scenegraph(predicted_graph, self.target)
        if completed_graph is None:
            print(f'No completed graph')
            return None
        completed_graph = self.assign_ids(completed_graph)
        if vlm:
            try:
                completed_graph = _reverse_coord_transform(copy.deepcopy(completed_graph))
            except:
                print("Error in reversing coordinate transformation: " , completed_graph)
                return None
        # vln_logger.info(f"Predicted scene graph: {completed_graph}", extra={"module": "SceneGraphPrediction"})

        # 3. Merge the predicted subgraph with original scene graph
        self.predicted_sg = completed_graph
        merged_graph = self.merge_subgraph(observed_graph, completed_graph)
        # vln_logger.info(f"Merged scene graph: {merged_graph}", extra={"module": "SceneGraphPrediction"})
        
        if vlm and merged_graph is not None:
            bev_image = visualize_BEV(occupancy_map, merged_graph, None, target_locations=frontier_locations, agent_location=agent_location)
            self.pred_bev_image = bev_image
        return merged_graph
    
    def plot_and_save(self, dump_dir, episode_label, step, llm_name, occupancy_map, frontier_locations, agent_coordinate, sub_scene_graph, predicted_sub_scene_graph):
        """ Runs the visualization function in a separate process to avoid blocking. """
        llm_name = '_'.join(llm_name.split(':'))
        save_path = f"{dump_dir}bev_{llm_name}/{episode_label}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.obs_bev_image is not None:
            bev_image_path = f"{save_path}/{step}_obs_bev.png"
            cv2.imwrite(bev_image_path, self.obs_bev_image[..., :3][..., ::-1])
        if self.pred_bev_image is not None:
            pred_bev_image_path = f"{save_path}/{step}_pred_bev.png"
            cv2.imwrite(pred_bev_image_path, self.pred_bev_image[..., :3][..., ::-1])

    # def predict_scenegraph(self, frontier_location, threshold=32, llm_name = "qwen2.5"):
    #     """
    #     Predict how the scene graph evolves conditioned on the visiting frontier by querying an LLM.
    #     """
    #     observed_scene_graph = self.scene_graph
    #     if not observed_scene_graph:
    #         return observed_scene_graph
    #     # print('Scene graph: ', observed_scene_graph)
    #     # 1. Prune the orginal graph to subgraph --> make prompt more compact
    #     sub_graph = self.prune_subgraph(deepcopy(observed_scene_graph), frontier_location, threshold)
    #     scene_description = json.dumps(sub_graph, indent=4)
    #     print('Sub graph: ', sub_graph)

    #     # 2. Query LLM to predict the observations and covert the text response to the graph
    #     llm_prompt = construct_query(self.prompts["SceneGraphComplementor"], target=self.target, frontier_location=frontier_location, scene_description=scene_description)
    #     # response = query_llm(llm_prompt, llm_name=llm_name) # qwen2.5:32b, deepseek-r1:14b, phi4
    #     # print('LLM response: ', response)
    #     # predicted_graph = format_graph(response)
    #     predicted_graph = sub_graph
    #     if predicted_graph is None:
    #         return sub_graph
    #     print('Pred graph: ', predicted_graph)

    #     try:
    #         completed_graph = self.complete_scenegraph(predicted_graph, self.target)
    #         print('Completed graph: ', completed_graph)
    #     except:
    #         completed_graph = sub_graph
    #         import ipdb; ipdb.set_trace()

    #     # 3. Merge the predicted subgraph with original scene graph
    #     merged_graph = self.merge_subgraph(observed_scene_graph, predicted_graph)
    #     print('Merged graph: ', merged_graph)

    #     # 4. Diff the scene graphs
    #     diff_graph = self.diff_scenegraph(sub_graph, predicted_graph)
    #     print('Diff graph: ', diff_graph)

    #     # visualize_scenegraph_diff_with_hierarchy_3d(sub_graph, predicted_graph, frontier_location, edge_threshold=50, save_path="scene_graph_hierarchy_diff_3d.png")
        
    #     # return merged_graph, diff_graph
    #     return sub_graph, predicted_graph, completed_graph, merged_graph, diff_graph

    def get_group_caption(self, prompt, objects, llm_name, vlm_name='llama3.2-vision', image=None, max_try_times=3):
        llm_prompt = construct_query(prompt, objects=objects)
        for _ in range(max_try_times):
            if image is not None:
                response = query_llm(llm_prompt, llm_name=vlm_name, image=image)
            else:
                response = query_llm(llm_prompt, llm_name=llm_name) # qwen2.5:32b, deepseek-r1:14b, phi4
            response = format_response(response)
            if len(response) != '' and ('answer' not in response) and len(response) < 30:
                break
        return response
    
    def generate_region_caption(self, scenegraph, target, llm_name = "qwen2.5", vlm_name='llama3.2-vision', image=None):
        for room_id, room in enumerate(scenegraph.get("rooms", [])):
            # Process regions in the room
            for region_id, region in enumerate(room.get("regions", [])):
                object_nodes = region.get("objects", [])
                object_captions = list(set([obj["caption"] for obj in object_nodes]))
                object_captions = json.dumps(object_captions).replace("\"",'')
                region["caption"] = self.get_group_caption(self.prompts["RegionCaptionGenerator"], object_captions, llm_name) if len(object_captions) != 0 else ''
                region["corr_score"] = self.get_text_sim_score(region.get("caption", ''), target)
                print('Objects: ', object_captions)
                print('Region: ', region["caption"])
        return scenegraph
    
    def generate_region_caption_withconf(self, scenegraph, target, llm_name = "qwen2.5", vlm_name='llama3.2-vision', image=None):
        for room_id, room in enumerate(scenegraph.get("rooms", [])):
            # Process regions in the room
            for region_id, region in enumerate(room.get("regions", [])):
                object_nodes = region.get("objects", [])
                object_captions = list(set(["{}:{:.2f}".format(obj["caption"], obj.get("confidence", 0.1)) for obj in object_nodes]))
                object_captions = json.dumps(object_captions).replace("\"",'')
                region["caption"] = self.get_group_caption(self.prompts["RegionCaptionGenerator_WithConfidence"], object_captions, llm_name) if len(object_captions) != 0 else ''
                region["corr_score"] = self.get_text_sim_score(region.get("caption", ''), target)
                print('Objects: ', object_captions)
                print('Region: ', region["caption"])
        return scenegraph
    
    def get_unlabelregion_description(self, sub_graph):
        region_description = ''
        regions = []
        for room in sub_graph["rooms"]:
            regions.extend(room.get("regions", []))
        count = 0
        for i, region in enumerate(regions):
            objects = ["{}:{:.2f}".format(obj["caption"], obj.get("confidence", 0.1)) for obj in region['objects']]
            if len(objects) > 0:
                region_description += f"Region {i} contained objects: {objects} \n"
                region['idx'] = count
                count += 1
        return region_description, sub_graph
    
    def get_global_group_caption(self, prompt, objects, llm_name, vlm_name='llama3.2-vision', image=None, max_try_times=3):
        llm_prompt = construct_query(prompt, objects=objects)
        for _ in range(max_try_times):
            if image is not None:
                response = query_llm(llm_prompt, llm_name=vlm_name, image=image)
            else:
                response = query_llm(llm_prompt, llm_name=llm_name) # qwen2.5:32b, deepseek-r1:14b, phi4
            response = format_graph(response)
            if response is not None:
                # print('Generated: ', response)
                break
        return response
    
    def generate_region_caption_global(self, scenegraph, target, llm_name = "qwen2.5", vlm_name='llama3.2-vision', image=None):
        region_description, scenegraph = self.get_unlabelregion_description(scenegraph)
        # print('Region description: ', region_description)
        region_captions = self.get_global_group_caption(self.prompts["RegionCaptionGenerator_Global"], region_description, llm_name)
        region_idx_dict = {v['idx']: k for k, v in enumerate(region_captions)}
        for room_id, room in enumerate(scenegraph.get("rooms", [])):
            # Process regions in the room
            for region_id, region in enumerate(room.get("regions", [])):
                region_idx = region.get('idx', -1)
                region["caption"] = region_captions[region_idx_dict[region_idx]]['caption'] if region_idx in region_idx_dict else ''
                object_captions = list(set(["{}:{:.2f}".format(obj["caption"], obj.get("confidence", 0.1)) for obj in region.get("objects", [])]))
                region["corr_score"] = self.get_text_sim_score(region.get("caption", ''), target)
                print('Objects: ', object_captions)
                print('Region: ', region["caption"])
        return scenegraph

    def get_region_vis_feat(self, region, image_history, topk=1):
        objects = region['objects']
        unique_obj_count = {}
        for obj in objects:
            for i, idx in enumerate(obj['image_idx']):
                unique_obj_count.setdefault(idx, {})
                unique_obj_count[idx].setdefault(obj['caption'], []).append(obj['conf'][i])
        image_coverage = Counter()
        for idx in unique_obj_count:
            for obj, confs in unique_obj_count[idx].items():
                image_coverage[idx] += max(confs)
        # region['image_coverage'] = image_coverage
        # region['unique_obj_count'] = unique_obj_count
        vis_feat = None
        coverage_sum = 0
        for image_idx, coverage in image_coverage.most_common(topk):
            image = image_history[image_idx]
            if vis_feat is None:
                vis_feat = coverage*self.get_vis_feat(image)
            else:
                vis_feat += self.get_vis_feat(image)
            coverage_sum += coverage
        vis_feat /= coverage_sum
        region['idx_list'] = list(image_coverage.keys())
        region['coverages'] = list(image_coverage.values())
        return vis_feat

    def generate_region_caption_TopK_clip(self, scene_graph, image_history, topk=3, img_topk=3):
        for room in scene_graph['rooms']:
            for region in room['regions']:
                similarities = []
                vis_feat = self.get_region_vis_feat(region, image_history, topk=img_topk)
                for caption in region_captions:
                    similarities.append(self.get_vis_sim_score(vis_feat, caption))
                captions = sorted(zip(similarities, region_captions), key=lambda x: x[0], reverse=True)[:topk]
                region['caption'] = {}
                region['clip_feat'] = vis_feat
                for similarity, caption in captions:
                    region['caption'][caption] = similarity
                if self.region_correlation == 'text':
                    region['corr_score'] = []
                    for caption, conf in region['caption'].items():
                        region['corr_score'].append(self.get_text_sim_score(caption, self.target))
                elif self.region_correlation == 'visual':
                    region['corr_score'] = self.get_vis_sim_score(vis_feat, self.target)
                elif self.region_correlation == 'cooccurrence':
                    region['corr_score'] = []
                    for caption, conf in region['caption'].items():
                        region['corr_score'].append(self.get_cooccurrence_region(caption, self.target))
        return scene_graph

    def generate_region_caption_TopK(self, scenegraph, target, llm_name = "qwen2.5", vlm_name='llama3.2-vision', image=None):
        if self.step - self.region_update_step >= 10:
            self.region_update_step = self.step
        else:
            return None
        vln_logger.increase_function_call_count('get_group_caption')
        # Output the full distribution over all the region candidates, or the top-k candidates
        region_description, scenegraph = self.get_unlabelregion_description(scenegraph)
        # print('Region description: ', region_description)
        region_captions = self.get_global_group_caption(self.prompts["RegionCaptionGenerator_TopK"], region_description, llm_name)
        try:
            region_idx_dict = {v['idx']: k for k, v in enumerate(region_captions)}
        except:
            return None
        f = open('region_caption.log', 'a')
        for room_id, room in enumerate(scenegraph.get("rooms", [])):
            # Process regions in the room
            for region_id, region in enumerate(room.get("regions", [])):
                region_idx = region.get('idx', -1)
                region["caption"] = region_captions[region_idx_dict[region_idx]]['caption'] if region_idx in region_idx_dict else ''
                object_captions = list(set(["{}:{:.2f}".format(obj["caption"], obj.get("confidence", 1.0)) for obj in region.get("objects", [])]))
                region["corr_score"] = get_text_sim_score(self.clip_model_list, region.get("caption", ''), target)[2]
                # print('Objects: ', object_captions)
                # print('Region: ', region["caption"])
                f.write(f"Objects: {object_captions}\n")
                f.write(f"Region: {region['caption']}\n")
        f.close()
        return scenegraph
    
    def get_gt_scenegrah(self, frontier_location, llm_name = "qwen2.5", threshold=40):
        if self.scene_graph_gt is None:
            scene_graph_gt = load_scene_graph(vln_logger.sim, agent=vln_logger.agent, distance_threshold=100.0, connected_thre=90)
            scene_graph_gt = self.complete_scenegraph(scene_graph_gt, self.target)
            scene_graph_gt = self.merge_objects(scene_graph_gt, distance_threshold=20.0, min_samples=1)
            self.scene_graph_gt = self.generate_region_caption(scene_graph_gt, self.target, llm_name='llama3.2-vision')
        sub_scene_graph = self.prune_subgraph(copy.deepcopy(self.scene_graph_gt), frontier_location, threshold=threshold)
    
        # # NOTE: ground truth scene graph and subgraph
        # Habitat = self.simulator._env._sim
        # ground_truth_scene_graph = load_scene_graph(Habitat)
        # location = [5.0, 10.0]  # example spatial location
        # distance_threshold = 10.0  # radius 
        # scene_graph_dict = get_subgraph(Habitat, location, distance_threshold)

        
        # path_scene = f"GT_scene_graphs/{self.experiment_name}/{str(scene_name)}/{str(self.count_episodes)}"
        # os.makedirs(path_scene, exist_ok=True)  
        # with open(f"{path_scene}/gt_scene_graph.json", "w") as f:
        #     f.write(ground_truth_scene_graph)
            
        # with open(f"{path_scene}/sub_scene_graph.json", "w") as f:
        #     f.write(scene_graph_dict) 
            
        # print(f"Saved Ground truth scene graph & subgraph at {path_scene}")
        return sub_scene_graph

    def set_gt_scenegrah_from_region_dict(self, region_list):
        # TODO: a hack to connect with current scnegraph, need to replace room level with floor level
        scene_graph_gt = {'rooms': [ region_list ]}
        self.scene_graph_gt = self.complete_scenegraph(scene_graph_gt, self.target)
        # sub_scene_graph = self.prune_subgraph(copy.deepcopy(self.scene_graph_gt), frontier_location, threshold=32)
        # return sub_scene_graph
