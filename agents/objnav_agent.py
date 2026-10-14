#!/usr/bin/env python3
import math
import time
import os
import datetime

import torch
import open3d as o3d

from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from PIL import Image
import yaml
import quaternion
from yacs.config import CfgNode as CN

import numpy as np
import cv2
import skimage.morphology

import open_clip
from utils.fmm_planner import FMMPlanner
from utils.mapping import (
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
    get_camera_K
)
from utils.model_utils import compute_clip_features
from utils.slam_classes import MapObjectList
from utils.detection_segmentation import Object_Detection_and_Segmentation
from utils.compute_similarities import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
    color_by_clip_sim,
    cal_clip_sim
)
from utils.imagine_nav_planner import ImagineNavPlanner
from constants import color_palette, category_to_id, categories_21, categories_21_plus_stairs, detector_classes, bg_classes
from utils.vis import draw_line, vis_result_fast, add_img, add_text, remove_image_border, resize_with_height, get_pyplot_as_numpy_img
from utils.explored_map_utils import (
    build_full_scene_pcd,
    detect_frontier,
)
from utils.utils_llm import query_llm, construct_target_query
import utils.pose as pu
from utils.scene_graph_utils import shift_gt_sg, plot_region
from utils.vln_logger import vln_logger

# import keyboard

# Disable torch gradient computation
torch.set_grad_enabled(False)

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

class ObjectNav_Agent(Agent):
    def __init__(self, args, follower=None) -> None:
        vln_logger.info("Initializing ObjectNav_Agent")
        # ------------------------------------------------------------------
        ##### Initialize basic config
        # ------------------------------------------------------------------
        self.args = args
        self.episode_n = 0
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        # self.device = "cuda:{}".format(self.args.gpu_id)
        self.device = torch.device("cuda")

        self.dump_dir = f"{args.dump_location}/{args.exp_name}/"

        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # ------------------------------------------------------------------
        ##### Initialize the perception model
        # ------------------------------------------------------------------

        # scene graph
        # self.exp_config = get_config(config_paths=["exp_configs/"+ self.args.exp_config])
        with open("exp_configs/"+ self.args.exp_config) as f:
            self.exp_config = yaml.safe_load(f)
        
        if self.args.gt_perception:
            self.classes = categories_21_plus_stairs
        else:
            # detection class varies depends on the dataset
            if self.args.task_config == 'objectnav_mp3d_rgbd.yaml':
                self.classes = categories_21_plus_stairs # mp3d has 21 categories
            else:
                self.classes = detector_classes 
        self.obj_det_seg = Object_Detection_and_Segmentation(self.args, self.classes, self.device)

        # Initialize the CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        self.clip_model = self.clip_model.to(self.device).half()
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        
        self.annotated_image = None


        # 3D mapping
        self.camera_K = get_camera_K(
            self.args.frame_width, self.args.frame_height, self.args.hfov)

        fcfg = open('configs/mapping_base.yaml')
        self.cfg = CN.load_cfg(fcfg)
        self.init_map_and_pose()

        # ------------------------------------------------------------------
        ##### Initialize navigation
        # ------------------------------------------------------------------
        if follower is not None:
            self.follower = follower
            
        self.text_queries = ''
        
        self.turn_angle = args.turn_angle
        self.init_map_and_navigation_param()
        # define global states
        self.action_keys = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN']
        # TODO add colors!
        gray_text = (128, 128, 128)
        red_text = (0, 0, 255)
        green_text = (0, 255, 0)
        blue_text = (255, 0, 0)
        self.state_dict = {
            'initial': ('just started', gray_text),
            'candidate': ('go to the candidate object', green_text),
            'reperception': ('go to the reperception candidate object', green_text),
            'imagine_nav': ('go to frontier with imagine nav planner', blue_text),
            'frontier_nav': ('go to frontier with max similarity', blue_text),
            'best_object': ('go to the nearest object with max similarity', blue_text),
            'random': ('walk randomly', gray_text),
            'upstair': ('go to the upstair', red_text),
            'downstair': ('go to the downstair', red_text),
            'another_floor': ('check if reached another floor', red_text),
            'keyboard': ('use keyboard to control the agent', red_text),
            'stair_entrance': ('go to the stair entrance', red_text),
            'climbing_stair': ('start climbing the stair', red_text),
            'new_floor': ('arrive at a new floor', red_text),
        }
        self.global_state = 'initial'
        
        ### for vidoe saving ###
        self.vis_frames = []
        # debug parameters
        self.use_keyboard = args.keyboard_actor

    def reset(self, episode_label, scene_id, episode_id) -> None:
        self.episode_n += 1
        self.episode_label = episode_label
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.init_map_and_pose()
        self.init_map_and_navigation_param()
        self.image_history = []
        ### for vidoe saving ###
        self.vis_frames = []
        self.metrics = {'distance_to_goal': 0, 'success': 0.0, 'spl': 0.0}
        vln_logger.reset()
    
    def update_metrics(self, metrics):
        self.metrics = metrics

    def init_map_and_pose(self):
        # local map
        self.map_size = self.args.map_size_cm // self.args.map_resolution
        self.map_real_halfsize  = self.args.map_size_cm / 100.0 / 2.0
        self.local_w, self.local_h = self.map_size, self.map_size

        self.reset_map()
        self.last_grid_pose = [self.map_size/2, self.map_size/2]
        self.last_real_pose = [0, 0]
        self.origins_grid = [self.map_size/2, self.map_size/2]
        self.origins_real = [0.0, 0.0]
        self.col_width = 1

        # stair maps
        # self.global_height_image = np.zeros((self.local_w, self.local_h))
        # self.global_height_gradient_img = np.zeros((self.local_w, self.local_h))
        # self.connected_img = np.zeros((self.local_w, self.local_h))
        # self.traversable_occupancy_map = np.zeros((self.local_w, self.local_h))
        # self.stair_map_img = np.zeros((self.local_w, self.local_h))
        # self.filtered_stair_map_img = np.zeros((self.local_w, self.local_h))
   
    def reset_map(self, camera_position=None):
        # single map
        self.explored_map = np.zeros((self.local_w, self.local_h))
        self.obstacle_map = np.zeros((self.local_w, self.local_h))
        self.visited_vis = np.zeros((self.local_w, self.local_h))
        self.goal_map = np.zeros((self.local_w, self.local_h))
        self.previous_goal_map = np.zeros((self.local_w, self.local_h))
        self.similarity_obj_map = np.zeros((self.local_w, self.local_h))
        self.similarity_img_map = np.zeros((self.local_w, self.local_h))
        self.collision_map = np.zeros((self.local_w, self.local_h))
        self.collision_map_final = np.zeros((self.local_w, self.local_h))
        self.collision_map_temporal = np.zeros((self.local_w, self.local_h))
        self.distance_to_goal_map = np.zeros((self.local_w, self.local_h))
        self.height_map = np.zeros((self.local_w, self.local_h))
        self.gradient_map = np.zeros((self.local_w, self.local_h))

        # merged map
        self.occupancy_map = np.zeros((self.local_w, self.local_h))
        self.semantic_map = np.zeros((self.local_w, self.local_h))
        self.traversible_map = np.zeros((self.local_w, self.local_h))

        if camera_position is not None:
            pcd = self.remove_diff_floor_points_cell(self.point_sum, camera_position)
            self.update_map(pcd, camera_position, self.args.map_height_cm / 100.0 /2.0)
        
    def move_map_and_pose(self, shift, axis):
        self.explored_map = pu.roll_array(self.explored_map, shift, axis) # need
        self.obstacle_map = pu.roll_array(self.obstacle_map, shift, axis) # need
        self.visited_vis = pu.roll_array(self.visited_vis, shift, axis)
        self.goal_map = pu.roll_array(self.goal_map, shift, axis)
        self.previous_goal_map = pu.roll_array(self.previous_goal_map, shift, axis)
        self.similarity_obj_map = pu.roll_array(self.similarity_obj_map, shift, axis)
        self.similarity_img_map = pu.roll_array(self.similarity_img_map, shift, axis)
        self.collision_map = pu.roll_array(self.collision_map, shift, axis) # need
        self.collision_map_final = pu.roll_array(self.collision_map_final, shift, axis)
        self.collision_map_temporal = pu.roll_array(self.collision_map_temporal, shift, axis)
        
        self.last_grid_pose = pu.roll_pose(self.last_grid_pose, shift, axis)
        self.origins_grid = pu.roll_pose(self.origins_grid, shift, axis)
        self.origins_real = pu.roll_pose(self.origins_real, -shift * self.args.map_resolution / 100.0, axis)

        if self.args.gt_scenegraph:
            shift_gt_sg(self.gt_scenegraph, shift, axis) # gt_scenegraph is python dictionary, and dictionary is pass by reference so mutable
        self.imagine_nav_planner.move_grid_origin(shift, axis)
    
    def reset_stair_climb_params(self, current_floor_camera_height):
        self.upstair_flag = False
        self.downstair_flag = False
        self.another_floor = False
        self.start_climbing = False
        self.found_stair_goal = False
        self.new_floor_pcd_num = 0
        self.height_diff = 0
        self.current_floor_camera_height = current_floor_camera_height
        self.found_stair_goal = False
        self.last_stair_goal_ind = None
        self.floor_height_threshold = self.args.FLOOR_HEIGHT
        self.early_stair_climb = False
        self.climbing_step = 0
        
    def init_map_and_navigation_param(self):
        
        # 3D mapping
        self.point_sum = o3d.geometry.PointCloud()
        self.init_sim_position = None
        self.init_sim_rotation = None
        self.init_agent_position = None
        self.Open3D_traj = []
        self.objects = MapObjectList(device=self.device)
        self.object_id_count = {'count': 0}
        self.nearest_point = None
        self.current_grid_pose = None
        
        self.relative_angle = 0
        self.eve_angle = 0

        # navigation
        self.l_step = 0
        self.step_times = []
        self.prev_goal_candidate = None
        self.curr_goal_candidate = None
        
        self.stuck_at_goal_count = 0
        self.no_frontiers_count = 0
        self.reach_goal_count = 0
        self.replan_count = 0
        self.collision_count = 0
        self.is_same_goal = False
        self.is_same_candidate = False
        self.dtg = 10000.
        self.min_distance_to_goal = 10000.
        self.dtg_history = [10000.]
        self.dtg_metric = 10000.

        self.is_running = True
        self.found_goal = False
        self.last_action = 0

        self.frontier_scores = None
        
        # stair related
        self.climbing_step = 0
        self.switch_upstair_count = 0
        self.switch_downstair_count = 0
        self.new_explored_area_each_step = []
        self.new_explored_area_in_last_T = 0
        self.exploration_scores = []
        self.reset_stair_climb_params(0)
        
        # prevent too many back and forth selection of two frontiers
        self.last_frontier_goal = None
        self.frontier_fmm_cost = -1

        self.clip_model_list = (self.clip_model, self.clip_preprocess, self.clip_tokenizer)
        self.imagine_nav_planner = ImagineNavPlanner(self.args, self.exp_config, self.clip_model_list)
        
        #load parameters. Is this an ok place to do it?
        self.trade_schedule = self.exp_config["utility"]["trade_off"].get("schedule",None)

    def act(self, observations: Observations, sim_info, step_data_queue, gui_prompt_queue=None, gui_viz_queue=None):
        time_step_info = 'Mapping time (s): \n'

        preprocess_s_time = time.time()
        # ------------------------------------------------------------------
        ##### At first step, get the object name and init the visualization
        # ------------------------------------------------------------------

        env, scene_name, episode_id, agent_state = sim_info['env'], sim_info['scene_name'], sim_info['episode_id'], sim_info['agent_state']
        if self.l_step == 0:



            self.init_sim_position = agent_state.sensor_states["depth"].position
            self.init_agent_position = agent_state.position
            self.init_sim_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

            if self.args.task_config == 'objectnav_mp3d_rgbd.yaml':
                cate_object = categories_21[observations['objectgoal'][0]] # mp3d has 21 classes
            else:
                cate_object = category_to_id[observations['objectgoal'][0]] # hm3d, ai2thor, hssd all has similar target
            
            # vln_logger.info(f"cate_object: {cate_object} | episode goal: {env.current_episode.object_category}")
            if self.args.gt_perception or self.args.task_config == 'objectnav_mp3d_rgbd.yaml':
                self.cate_object = cate_object
                self.text_queries = cate_object
            else:
                if cate_object == 'sofa':
                    cate_object = 'couch'
                if cate_object == 'tv_monitor':
                    cate_object = 'tv'
                # if cate_object == 'tv' or cate_object == 'plant' or cate_object == 'chair' or cate_object == 'toilet':
                #     self.text_queries = "looks like a " + cate_object
                self.text_queries = cate_object
                self.cate_object = cate_object
            
            # init logger
            vln_logger.new_image(450, 800+800)
            vln_logger.set_experiment_name(self.exp_config['experiment_name'])  
            vln_logger.set_agent({'position': env.current_episode.start_position, 'rotation': env.current_episode.start_rotation})
            vln_logger.set_sim(env._sim)
            vln_logger.set_scene(str(scene_name))
            vln_logger.set_episode(episode_id)
            vln_logger.set_dump_dir(self.dump_dir)
            
            # update scene graph
            self.imagine_nav_planner.set_obj_goal(self.cate_object)

        # if self.args.debug and self.l_step == 5:
        #     return 0 # end this episdoe
        
        # get gt informations
        # if self.args.gt_perception:
        self.goal_poses_metric = observations['goal_poses_metric']
        ##### debug: get the ground truth region with labels #####
        if self.args.gt_scenegraph:
            # TODO: make copy here, remember to shift the gt scene graph
            self.gt_scenegraph = observations['gt_scenegraph']
        
        self.imagine_nav_planner.set_step(self.l_step, self.episode_label)
        vln_logger.set_step(self.l_step)   
        
        # print("current position: ", agent_state.sensor_states["depth"].position)
        # ------------------------------------------------------------------
        ##### Preprocess the observation
        # ------------------------------------------------------------------

        image_rgb = observations['rgb']
        depth = observations['depth']
        image = transform_rgb_bgr(image_rgb) 
        self.annotated_image = image
        
        if self.args.gt_perception:
            semantic_img = observations['semantic']
            if self.args.save_perception_results:
                vln_logger.set_semantic(semantic_img)
            
            get_results, detections = observations['gt_detections']['get_results'], observations['gt_detections']['detections']

            if get_results is None:
                raise ValueError("get_results is None. Probably the scene doesn't have gt annotations.")
                
            if self.args.debug:
                from PIL import Image
                from habitat_sim.utils.common import d3_40_colors_rgb
                semantic_rgb = Image.new("P", (semantic_img.shape[1], semantic_img.shape[0]))
                semantic_rgb.putpalette(d3_40_colors_rgb.flatten())
                semantic_rgb.putdata((semantic_img.flatten() % 40).astype(np.uint8))
                semantic_rgb = semantic_rgb.convert("RGBA")
                semantic_rgb.save(f'{self.dump_dir}/current_semantic_process{self.args.process_id}.png')
                # debug to see gt detection
                # annotated_image  = vis_result_fast(image, detections, self.classes)
                # cv2.imwrite(f'{self.dump_dir}episodes/current_semantic_process{self.args.process_id}.png', annotated_image)
        else:
            get_results, detections = self.obj_det_seg.detect(image, image_rgb, self.classes)
        
        ##### log perception benchmakr #####
        if self.args.save_perception_results:
            vln_logger.set_rgb(image)
            vln_logger.set_depth(depth)
            vln_logger.add_detection(detections)
            
        clip_s_time = time.time()
        image_crops, image_feats, current_image_feats = compute_clip_features(
            image_rgb, detections, self.clip_model, self.clip_preprocess, self.device)
        
        # print('clip: %.3fs'%(clip_e_time - clip_s_time)) 
        
        self.image_history.append(image_rgb)
        if get_results:
            detection_results = {
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": self.classes,
                "image_crops": image_crops,
                "image_feats": image_feats,
                "token_logits": detections.confidence,
                "image_rgb": image_rgb,
                "idx": self.l_step
            }
        else:
            detection_results = None
        
   
        preprocess_e_time = time.time()
        time_step_info += 'Preprocess time:%.3fs\n'%(preprocess_e_time - preprocess_s_time)

        # ------------------------------------------------------------------
        ##### Object Set building
        # ------------------------------------------------------------------
        v_time = time.time()

        cfg = self.cfg
        depth = self._preprocess_depth(depth)
        
        camera_matrix_T = self.get_transform_matrix(agent_state)
        vln_logger.add_pose(camera_matrix_T)
        camera_position = camera_matrix_T[:3, 3]
        self.camera_position = camera_position
        self.Open3D_traj.append(camera_matrix_T)
        self.relative_angle = round(np.arctan2(camera_matrix_T[2][0], camera_matrix_T[0][0])* 57.29577951308232 + 180)
        # print("self.relative_angle: ", self.relative_angle)
        
        # convert detection_results to objects
        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = image_rgb,
            depth_array = depth,
            cam_K = self.camera_K,
            idx = self.l_step,
            gobs = detection_results,
            trans_pose = camera_matrix_T,
            bg_classes = bg_classes,
            class_names = self.classes,
        ) # point clouds in global frame

        
        obj_time = time.time()
        objv_time = obj_time - v_time
        # print('build objects: %.3fs'%objv_time) 
        time_step_info += 'build objects time:%.3fs\n'%(objv_time)

        if len(fg_detection_list) > 0 and len(self.objects) > 0 :
            spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
            visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
            agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)

            # Threshold sims according to cfg. Set to negative infinity if below threshold
            agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')
            self.objects.set_updated_at(self.l_step)
            self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, agg_sim, self.object_id_count)

        # Perform post-processing periodically if told so
        if cfg.denoise_interval > 0 and (self.l_step+1) % cfg.denoise_interval == 0:
            self.objects = denoise_objects(cfg, self.objects)
        if cfg.merge_interval > 0 and (self.l_step+1) % cfg.merge_interval == 0:
            self.objects = merge_objects(cfg, self.objects)
        if cfg.filter_interval > 0 and (self.l_step+1) % cfg.filter_interval == 0:
            self.objects = filter_objects(cfg, self.objects)
            
        sim_time = time.time()
        sim_obj_time = sim_time - obj_time 
        # print('calculate merge: %.3fs'%sim_obj_time) 
        time_step_info += 'calculate merge time:%.3fs\n'%(sim_obj_time)

        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                fg_detection_list[i]['global_id'] = self.object_id_count['count']
                self.object_id_count['count'] += 1
                self.objects.append(fg_detection_list[i])

        if self.args.debug:
            pass
            # print("------------------------------objects----------------------------")
            # for obj in self.objects:
            #     print(f"obj: {obj['class_name']}, lambda: {obj['bki_ft'][-1]}, variance: {obj['variance']}, num_det: {obj['num_detections']}, clip_norm: {obj['clip_ft'].norm()}")
        # ------------------------------------------------------------------
        ##### 2D Obstacle Map
        # ------------------------------------------------------------------
        f_map_time = time.time()
        
        local_grid_pose = [camera_position[0]*100/self.args.map_resolution + int(self.origins_grid[0]), 
                      camera_position[2]*100/self.args.map_resolution + int(self.origins_grid[1])] # x pointing up, z pointing right in image
        pose_x = int(local_grid_pose[0]) if int(local_grid_pose[0]) < self.map_size-1 else self.map_size-1
        pose_y = int(local_grid_pose[1]) if int(local_grid_pose[1]) < self.map_size-1 else self.map_size-1
        
        # Adjust the centriod of the map when the robot move to the edge of the map
        if pose_x < 100:
            self.move_map_and_pose(shift = 100, axis=0)
            pose_x += 100
        elif pose_x > self.map_size - 100:
            self.move_map_and_pose(shift = -100, axis=0)
            pose_x -= 100
        elif pose_y < 100:
            self.move_map_and_pose(shift = 100, axis=1)
            pose_y += 100
        elif pose_y > self.map_size - 100:
            self.move_map_and_pose(shift = -100, axis=1)
            pose_y -= 100
        
        self.current_grid_pose = [pose_x, pose_y]
        
        # visualize trajectory
        self.visited_vis = draw_line(self.last_grid_pose, self.current_grid_pose, self.visited_vis)
        self.last_grid_pose = self.current_grid_pose
        
        # Collision check
        self.collision_check(camera_position)
        full_scene_pcd = build_full_scene_pcd(depth, image_rgb, self.args.hfov)

        # build 3D pc map
        full_scene_pcd.transform(camera_matrix_T)
        self.point_sum += self.remove_full_points_cell(full_scene_pcd, camera_position) # filter out higher points
        
        if self.args.new_mapping:
            ###### new method that may #####
            obs_i_values, obs_j_values = self.pcd_to_grid_indx(full_scene_pcd)
            self.update_map_new(full_scene_pcd, camera_position, self.args.map_height_cm / 100.0 /2.0)
        else:
            ##### old method that may breake the map #####
            obs_i_values, obs_j_values = self.update_map(full_scene_pcd, camera_position, self.args.map_height_cm / 100.0 /2.0)

        # build height map and gradient map
        all_points = np.asarray(self.point_sum.points) 
        height_min = camera_position[1] - 1.5
        self.height_map, self.gradient_map = self.create_global_height_map(all_points, height_min)

        # detect frontier
        target_fmm_cost, frontier_edge_map, frontier_candidate_list = detect_frontier(self.explored_map, self.obstacle_map, self.current_grid_pose, threshold_point=8)
        self.frontier_edge_map = frontier_edge_map
        if len(frontier_candidate_list) == 0:
            self.no_frontiers_count += 1

        # print('voxel map: %.3fs'%(v_map_time - f_map_time)) 
        
        # ------------------------------------------------------------------
        ##### Calculate the similarities 
        # ------------------------------------------------------------------
            
        if gui_prompt_queue is not None and not gui_prompt_queue.empty():
            text_input, self.is_running = gui_prompt_queue.get()
            if text_input is not None:
                self.text_queries = text_input
            # print("self.text_queries: ", self.text_queries)

        clip_time = time.time()
        candidate_objects = []
        reperception_objects = []
        candidate_id = []
        self.upstair_candidate_objects = []
        self.downstair_candidate_objects = []
        similarity_threshold = 0.27

        synonym = ""
        if "toilet" in self.text_queries:
            similarity_threshold = 0.28
        if "chair" in self.text_queries or "plant" in self.text_queries:
            similarity_threshold = 0.26
        if "sofa" in self.text_queries:
            similarity_threshold = 0.285
        if "plant" in self.text_queries:
            synonym = "potted plant"
        similarities = None
        unvisited_objects = [obj for obj in self.objects if not obj['visited']]
        if len(self.objects) > 0:
            self.objects, similarities = color_by_clip_sim(self.text_queries, 
                                                        self.objects, 
                                                        self.clip_model, 
                                                        self.clip_tokenizer)
            
            similarities = similarities.cpu().numpy()
            
            self.objects, stairs_similarities = color_by_clip_sim('stairs', 
                                                        self.objects, 
                                                        self.clip_model, 
                                                        self.clip_tokenizer,
                                                        color_set = False)
            if self.args.gt_perception:
                candidate_objects = [self.objects[i] for i in range(len(self.objects)) 
                                    if self.objects[i]['caption'] in self.text_queries]
                reperception_objects = [] # no intermediate option
            else:
                candidate_objects = []
                reperception_objects = []
                for i in range(len(self.objects)):
                    is_similarity_high = similarities[i] > similarity_threshold
                    is_detection_confidence_high = max(self.objects[i]['conf']) > 0.85
                    has_multiple_observations = self.objects[i]['num_detections'] > 2
                    class_name = self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])]
                    is_target_object = class_name in self.text_queries or class_name in synonym

                    if is_target_object and (is_similarity_high or is_detection_confidence_high) and has_multiple_observations:
                        if self.objects[i]['llm_verified'] or self.objects[i]['llm_verified'] is None:                            
                            candidate_objects.append(self.objects[i])
                    if is_target_object and (is_similarity_high or is_detection_confidence_high) and not self.objects[i]['reperception_visited']:
                        if self.objects[i]['llm_verified'] or self.objects[i]['llm_verified'] is None:
                            reperception_objects.append(self.objects[i])

            # store highest or lowest point of the stairs in candidate list
            #################################################
            ##### TODO: modify stair candidate criteria #####
            #################################################
            for i in range(len(self.objects)):
                # if stairs_similarities[i] > 0.24 and self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])] == "stairs":
                    # all_stair_candidates.append(self.objects[i]['pcd']) # TODO: store all pcd of the stairs
                    # if self.objects[i]['bbox'].min_bound[1] > camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT + 0.2: # -0.8
                    # use point cloud instead 
                if hasattr(self.objects[i], 'caption'):
                    obj_name = self.objects[i]['caption']
                else:
                    obj_name = self.objects[i]['class_name'][np.argmax(self.objects[i]['conf'])]
                if stairs_similarities[i] > 0.24 and obj_name == "stairs":
                    average_point_height = np.mean(np.array(self.objects[i]['pcd'].points)[:, 1])
                    if average_point_height > camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT:
                        max_index = np.argmax(np.asarray(self.objects[i]['pcd'].points)[:, 1])
                        self.upstair_candidate_objects.append(self.objects[i]['pcd'])
                    else:
                        np.argmin(np.asarray(self.objects[i]['pcd'].points)[:, 1])
                        self.downstair_candidate_objects.append(self.objects[i]['pcd'])
                   
                if similarities[i] > 0.24:
                    self.similarity_obj_map[self.object_map_building(self.objects[i]['pcd'])] = similarities[i]
                # TODO: use same method to build map for stairs 
            
            
        image_clip_sim = cal_clip_sim("something near the " + self.text_queries, current_image_feats, self.clip_model, self.clip_tokenizer)
        # self.similarity_img_map[self.obstacle_map==1] = image_clip_sim.cpu().numpy() # Should not use self.obstacle_map becuase it will set everything to the current image similarity
        self.similarity_img_map[obs_i_values, obs_j_values] = image_clip_sim.cpu().numpy()
        f_clip_time = time.time()
        # print('calculate clip sim: %.3fs'%(f_clip_time - clip_time)) 
        time_step_info += 'calculate clip sim time:%.3fs\n'%(f_clip_time - clip_time)
        
        # NOTE junzhe: confidence of the object "being" the target or the object being "near" the target
        # TODO shall we use this to calculate exploitation score for objects?
        similarity_map = np.max(np.stack([self.similarity_obj_map, self.similarity_img_map]), axis=0)
        # self.save_similarity_map(similarity_map)

        self.merge_map(update_occupancy_map=True)
        self.global_bev_rgb_map = self.get_global_bev_rgb_map(camera_position)

        # ------------------------------------------------------------------
        ##### scene graph update
        # ------------------------------------------------------------------
        if self.args.imagine_nav_planner:
            self.imagine_nav_planner.update_scene_graph(self.objects, self.origins_grid, self.gradient_map, self.occupancy_map, self.image_history)
        
        # ------------------------------------------------------------------
        ##### frontier selection and determine goal location
        # ------------------------------------------------------------------

        # Note junzhe:
        # 1. check if goal is found in candidate_objects
        # 2. or, check if goal is found in reperception_objects
        # 3. or, set goal to frontier with max similarity or scene graph
        # 4. or, set goal to the object with max similarity,
        #        go to another floor if no frontier for 5 steps
        # 5. or, if there is no objects at all, walk randomly

        self.goal_map = np.zeros((self.local_w, self.local_h))

        self.found_goal = len(candidate_objects) > 0
        self.is_same_candidate = False
        self.curr_goal_candidate = None
        self.frontier_fmm_cost = 1000
        if len(candidate_objects) > 0:
            # 1. choose target object with max confidence
            if self.global_state != 'candidate':
                # only clean the goal map when there was no candidate before,
                # otherwise stack the candidates in the goal map
                self.goal_map = np.zeros((self.local_w, self.local_h))
            
            # TODO: use llm to repercieve the current candidate 
            # If current candidate is not target, then query the next candidate
            goal_candidate = None
            if self.args.llm_reperception and not self.args.gt_perception:
                max_conf = [max(obj['conf']) for obj in candidate_objects]
                sorted_conf_index = np.argsort(max_conf) # ascending order
                for i, index in enumerate(reversed(sorted_conf_index)): # descending order
                    current_candidate = candidate_objects[index]
                    if current_candidate['llm_verified'] is None:
                        # TODO: perform llm verification
                        prompt, prompt_image = construct_target_query(current_candidate, self.cate_object, self.image_history)
                        
                        if self.args.debug:
                            print("[LLM Reperception]: Verify candidate object")
                            # print("[LLM Reperception] Prompt Image: ", prompt_image)
                        for i in range(3): # query at most 3 times to get the correct format
                            query_result = query_llm(prompt, self.args.reperception_llm, prompt_image) # TODO: perform llm query, True or False
                            # query_result = format_response(query_result)
                            if 'yes' in query_result:
                                query_result = True
                                if self.args.debug:
                                    print("[LLM Reperception] Yes, this is the target object")
                                break
                            elif 'no' in query_result:
                                query_result = False
                                if self.args.debug:
                                    print("[LLM Reperception] No, this is not the target object")
                                break
                            else:
                                query_result = None
                            
                        current_candidate['llm_verified'] = query_result
                        if query_result:
                            goal_candidate = current_candidate
                            break
                    elif current_candidate['llm_verified'] is False:
                        continue # don't use this candidate
                    else: # llm_verified == True
                        goal_candidate = current_candidate
                        break # use this candidate
            else:
                # choose the candidate with max confidence
                max_conf = [max(obj['conf']) for obj in candidate_objects]
                max_index = np.argmax(max_conf)
                goal_candidate = candidate_objects[max_index]
                
            self.curr_goal_candidate = goal_candidate
            self.is_same_candidate = self.prev_goal_candidate is self.curr_goal_candidate
            
            if goal_candidate is not None:
                self.global_state = 'candidate'
                goal_candidate_pcd = goal_candidate['pcd']
                i_values, j_values = self.object_map_building(goal_candidate_pcd)
                self.goal_map[i_values, j_values] = 1
                self.nearest_point = self.find_nearest_point_cloud(goal_candidate_pcd, camera_position)
                
                diff = np.array(self.nearest_point) - np.array(camera_position)
                self.dtg_metric = np.linalg.norm([diff[0], diff[2]])
                # Replace the nearest point with the edge pixel 
                if self.args.edge_goal:
                    # TODO: do we want to update it every time? Or only give it once when the goal is found?
                    # points = np.hstack((i_values.reshape(-1, 1), j_values.reshape(-1, 1))).T
                    edge_i_values, edge_j_values = self.find_goal_on_obstacle_edge(self.nearest_point)
                    if edge_i_values is not None and edge_j_values is not None:
                        self.goal_map[edge_i_values, edge_j_values] = 1
                    else:
                        vln_logger.info("No edge candidate point found")
                
        elif len(reperception_objects) > 0:
            # 2. choose target object with max confidence
            self.global_state = 'reperception'
            max_conf = [max(obj['conf']) for obj in reperception_objects]
            max_index = np.argmax(max_conf)
            self.reperception_object = reperception_objects[max_index]
            self.goal_map[self.object_map_building(self.reperception_object['pcd'])] = 1
            self.nearest_point = self.find_nearest_point_cloud(self.reperception_object['pcd'], camera_position)
        elif len(frontier_candidate_list) > 0:
            self.no_frontiers_count = 0
            best_frontier_id = None
            frontier_scores = {}

            # 3.2 choose frontier with max similarity
            self.global_state = 'frontier_nav'
            simi_max_score = []
            for i in range(len(frontier_candidate_list)):
                fmb = self.get_frontier_boundaries((frontier_candidate_list[i][0], 
                                                frontier_candidate_list[i][1]),
                                                (self.local_w/12, self.local_h/12),
                                                (self.local_w, self.local_h))

                cropped_sim_map = similarity_map[fmb[0]:fmb[1], fmb[2]:fmb[3]]
                simi_max_score.append(np.max(cropped_sim_map))

            best_frontier_id = 0
            if len(simi_max_score) > 0:
                if max(simi_max_score) > 0.22:
                    best_frontier_id = simi_max_score.index(max(simi_max_score))
            frontier_scores.update({'baseline': simi_max_score})

            # 3.1 choose frontier with imagine nav planner
            if self.args.imagine_nav_planner:
                self.global_state = 'imagine_nav'
                # print(f"frontier number: {len(frontier_candidate_list)}")
                if self.args.gt_scenegraph:
                    # determine which floor the agent is in 
                    floor_avg_heights = [floor['floor_avg_height'] for floor in self.gt_scenegraph['floors']]
                    floor_id = np.argmin(np.abs(np.array(floor_avg_heights) - self.camera_position[1]))
                start_time = time.time()
                scores, best_frontier_id, self.exploration_scores = self.imagine_nav_planner.fbe(
                    frontier_candidate_list,
                    self.current_grid_pose,
                    self.traversible_map,
                    self.occupancy_map,
                    self.global_bev_rgb_map,
                    self.gt_scenegraph['floors'][floor_id] if self.args.gt_scenegraph else None,
                )
                # do exploration scheduling; change weight dynamically.
                if(self.trade_schedule is not None):
                    dst_w,expt_w,expr_w = self.imagine_nav_planner.with_distance, self.imagine_nav_planner.with_exploitation, self.imagine_nav_planner.with_exploration # copy the weights
                    #naive scheduling (no weights)
                    weighted_exploration,weighted_exploitation,weighted_distance = scores['exploration']*expr_w,scores['exploitation']*expt_w,scores['distance']*dst_w
                if(self.trade_schedule == 'basic'):

                    exploitation_only = weighted_exploitation+weighted_distance 

                    best_exploitation_id = np.argmax(exploitation_only)
                    if(self.l_step<80 or exploitation_only[best_exploitation_id]<0.2):
                        best_frontier_id = np.argmax(weighted_exploration+weighted_distance) #exploration and distance
                    else:
                        best_frontier_id = best_exploitation_id
                
                if(self.trade_schedule == 'sigmoid'):
                    exploitation_gain = 1.0/(1+np.exp(-0.04*(self.l_step-120)))
                    exploration_gain = 1-exploitation_gain

                    best_frontier_id = np.argmax(weighted_exploration*exploration_gain+weighted_exploitation*exploitation_gain+weighted_distance)

                frontier_scores.update(scores)
                if self.args.debug:
                    print("imagine nav planner fbe time: ", time.time()-start_time)
                
                # determine early_climb criteria
                all_frontier_explore_small = (self.exploration_scores < 0.1).all()
                T_prev = max(self.l_step - 20, 0)
                self.new_explored_area_in_last_T = np.sum(self.new_explored_area_each_step[T_prev:-1])
                explored_area_small = self.new_explored_area_in_last_T < 100 # smaller than 50 pixels
                if all_frontier_explore_small and explored_area_small and self.args.early_climb and self.l_step > 20:
                    self.early_stair_climb = True
                    
            best_frontier = frontier_candidate_list[best_frontier_id]
            self.goal_map[best_frontier[0], best_frontier[1]] = 1

            self.frontier_scores = frontier_scores
            self.best_frontier_id = best_frontier_id
            self.frontier_candidate_list = frontier_candidate_list
            self.frontier_fmm_cost = target_fmm_cost[best_frontier_id]
        elif len(unvisited_objects) > 0:
            # 4. choose target object with max similarity
            self.global_state = 'best_object'
            unvisited_similarities = [sim for obj, sim in zip(self.objects, similarities) if not obj.get('visited', False)]
            max_index = np.argmax(unvisited_similarities)
            self.best_object = unvisited_objects[max_index]
            self.goal_map[self.object_map_building(self.best_object['pcd'])] = 1
            self.nearest_point = self.find_nearest_point_cloud(self.best_object['pcd'], camera_position)
            ###############################################
            ##### TODO: modify criteria to go upstair #####
            ###############################################
            if self.no_frontiers_count > 5:
                ##### if not found goal and always no frontier, go to another floor #####
                self.another_floor = True
        else:
            # 5. walk randomly
            self.global_state = 'random'
            goal_pose_x = int(np.random.rand() * self.map_size)
            goal_pose_y = int(np.random.rand() * self.map_size)
            self.goal_map[goal_pose_x, goal_pose_y] = 1
            
        ###################################################################################
        ##### if going to another floor, using stiar climbing agent to overwirte goal #####
        ###################################################################################
        if self.l_step < 40 and not self.args.no_deal_stair_spawn:
            # regardless of found goal or not, during early steps, if we are on stairs, we should give up goal and go to another floor first. 
            if self.no_frontiers_count > 5:
                self.another_floor = True
            
        # determine going up or going down 
        if (self.another_floor or self.early_stair_climb) and (not self.upstair_flag and not self.downstair_flag):
            if len(self.upstair_candidate_objects) > 0 and len(self.downstair_candidate_objects) == 0:
                self.upstair_flag = True
            elif len(self.upstair_candidate_objects) == 0 and len(self.downstair_candidate_objects) > 0:
                self.downstair_flag = True
            elif len(self.upstair_candidate_objects) > 0 and len(self.downstair_candidate_objects) > 0:
                if self.text_queries == "bed":
                    self.upstair_flag = True # NOTE: Might need to remove this heuristic
                elif len(self.upstair_candidate_objects) >= len(self.downstair_candidate_objects):
                    self.upstair_flag = True 
                else:
                    self.downstair_flag = True
                        
        # TODO: naive switch, need to cut off other goal selection as well
        if (self.another_floor or self.early_stair_climb) and (self.upstair_flag or self.downstair_flag) and not self.args.no_stair_climbing:

            # disable found goal
            self.found_goal = False

        # if self.another_floor and (self.upstair_flag or self.downstair_flag) and (not self.found_goal or self.start_climbing) and not self.args.no_stair_climbing:
            if self.climbing_step > 20 and not (camera_position[1] - self.current_floor_camera_height > 0.5):
                # NOTE: A naive failure handling idea, if no height increase after 10 steps of climbing, restart from finding stair entrance
                self.start_climbing = False
                self.climbing_step = 0
            start = time.time()
            stair_list = self.upstair_candidate_objects if self.upstair_flag else self.downstair_candidate_objects
            self.stair_goal_ind, self.stair_obstacle_map, self.stair_traversible_map, self.stair_goal_map = self.stair_climbing_policy(stair_list, camera_position)
            
            if self.start_climbing:
                self.global_state = 'climbing_stair'
            elif self.found_stair_goal:
                self.global_state = 'stair_entrance'
            if self.args.debug:
                print("Stair calc time: ", time.time()-start)
            
        # NOTE seems that this part is only useful for cheat planner
        # NOTE go to the goal if found, otherwise walk randomly
        if np.sum(self.goal_map)>1:
            x = self.nearest_point[0]
            y = self.nearest_point[1]
            z = self.nearest_point[2]
        elif np.sum(self.goal_map)==1:
            f_pos = np.argwhere(self.goal_map == 1)
            stg = f_pos[0]
            x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            y = camera_position[1]
            z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
        else:
            vln_logger.warning("Goal map is empty, walking randomly...")
            # stg = self._get_closed_goal(self.obstacle_map, self.last_grid_pose, find_big_connect(self.goal_map))
            goal_pose_x = int(np.random.rand() * self.map_size)
            goal_pose_y = int(np.random.rand() * self.map_size)
            self.goal_map[goal_pose_x, goal_pose_y] = 1
            f_pos = np.argwhere(self.goal_map == 1)
            stg = f_pos[0]
            
            x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            y = camera_position[1]
            z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
        Open3d_goal_pose = [x, y, z] # NOTE: seems that this variable is only used by cheat planner
            
        self.merge_map(update_traversible_map=True, update_semantic_map=True)

        # ------------------------------------------------------------------
        ##### Path planning and action selection
        # ------------------------------------------------------------------
        
        if not self.args.fmm_planner and not self.args.imagine_nav_planner:
            ########################################################
            ##### TODO: need to be removed, cheating planner  ######
            ########################################################
            Open3d_goal_pose = [x, y, z] # NOTE: seems that this varialbe is only used by cheat planner
            
            # handle stairs
            if self.another_floor and not self.found_goal:
                # NOTE: cheating goal selection for stairs
                # note junzhe: find highest or lowest stair
                if self.upstair_flag and len(self.upstair_candidate_objects) > 0:
                    self.global_state = 'upstair'
                    highest_stair = max(self.upstair_candidate_objects, key=lambda coord: coord[1]) 
                    Open3d_goal_pose = [highest_stair[0], highest_stair[1]+0.8, highest_stair[2]]
                    # TODO: use same method to build map for stairs 
                elif self.downstair_flag and len(self.downstair_candidate_objects) > 0:
                    self.global_state = 'downstair'
                    lowest_stair = min(self.downstair_candidate_objects, key=lambda coord: coord[1]) 
                    Open3d_goal_pose = [lowest_stair[0], lowest_stair[1]+0.8, lowest_stair[2]]
                    # TODO: use same method to build map for stairs 
                else:
                    self.global_state = 'another_floor'
                    self.reset_map(camera_position)
                    new_floor_pcd = self.detect_plane_points_cell(full_scene_pcd, camera_position)
                    if len(new_floor_pcd.points) > 1500:
                        ground_pcd = self.plane_segmentation_xy(new_floor_pcd)
                        print(len(ground_pcd.points))
                        if len(ground_pcd.points) > 1500:
                            self.another_floor = False
                            
                            objects_to_keep = []
                            for obj in self.objects:
                                if abs(obj['bbox'].get_center()[1] - camera_position[1])  < 1.0 or 'stairs' in obj['class_name']:
                                    objects_to_keep.append(obj)
                            self.objects = MapObjectList(objects_to_keep)
            
            Rx = np.array([[0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0]])
            R_habitat2open3d = self.init_sim_rotation @ Rx.T
            self.habitat_goal_pose = np.dot(R_habitat2open3d, Open3d_goal_pose) + self.init_agent_position
            habitat_final_pose = self.habitat_goal_pose.astype(np.float32)

            # NOTE junzhe: cheat here, search_navigable_path returns shortest path according to the mesh
            plan_path = []
            plan_path = self.search_navigable_path(
                habitat_final_pose
            )
            if len(plan_path) > 1 and self.another_floor:
                vector = plan_path[-1] - plan_path[-2]
                one_more_point = (vector / np.linalg.norm(vector)) * 0.5 + plan_path[-1]
                one_more_point[1] = plan_path[-1][1]
                # if one_more_point[1] >= self.Open3d_goal_pose_old[1]:
                # print("init agent pos:", self.init_agent_position)
                # print("goal prev:", self.habitat_goal_pose)
                self.habitat_goal_pose = one_more_point
                # print("goal new:", self.habitat_goal_pose)
                plan_path = self.search_navigable_path(
                    one_more_point
                )
                    # self.Open3d_goal_pose_old = self.habitat_goal_pose
                # else:
                #     plan_path = self.search_navigable_path(
                #         self.Open3d_goal_pose_old
                #     )
                #     self.habitat_goal_pose = self.Open3d_goal_pose_old
        
            if len(plan_path) > 1:
                # NOTE: cheat planning
                plan_path = np.dot(R_habitat2open3d.T, (np.array(plan_path) - self.init_agent_position).T).T
                action = self.greedy_follower_act(plan_path)
            ########################################################
            ##### TODO: need to be removed, cheating planner  ######
            ########################################################
        
        else:
            # plan a path by fmm
            self.stg, self.fmm_stop, fmm_path = self._get_stg()
            fmm_path = np.array(fmm_path) 
            fmm_path_x = (fmm_path[:, 0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            fmm_path_y = fmm_path[:, 0] * 0
            fmm_path_z = (fmm_path[:, 1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
            self.fmm_path = fmm_path.astype(np.int32)

            plan_path = np.stack((fmm_path_x, fmm_path_y, fmm_path_z), axis=-1)
            action = self.fmm_act()
        
        # use keyboard to control the agent
        if self.use_keyboard:
            self.global_state = 'keyboard'
            action = self.keyboard_act(action)
            # print("z_angle: ", self.eve_angle)
            
        # ------------------------------------------------------------------
        ##### Stop and Stuck criterion
        # ------------------------------------------------------------------
        planner = FMMPlanner(np.ones((self.local_w, self.local_h)))
        planner.set_multi_goal(self.goal_map)
        dtg = planner.fmm_dist[self.current_grid_pose[0], self.current_grid_pose[1]]
        dtg_obstacle = self.fmm_dist[self.current_grid_pose[0], self.current_grid_pose[1]]

        # found goal when not climbing the stair
        if self.found_goal and not self.start_climbing:
            # TODO: is it better if compare the last candidate goal and current candidate goal?
            is_same_goal_map = np.abs(self.goal_map - self.previous_goal_map).sum() <= 8 if self.previous_goal_map is not None else False # minimum pcd change
            is_same_goal = self.is_same_candidate or is_same_goal_map # becuase Edge Goal updates every timestep. 
        else:
            is_same_goal = np.abs(self.goal_map - self.previous_goal_map).sum()==0 if self.previous_goal_map is not None else False
        
        ##### adjusted distance to goal: lower hm3d score #####
        # self.min_distance_to_goal = min(self.dtg_history)
        is_stuck_at_goal = dtg_obstacle >= self.min_distance_to_goal - 0.01
        # is_near_goal = dtg < self.args.NEAR_GOAL_DISTANCE and len(self.fmm_path) < 3 # very close or no path
        ##### adjusted distance to goal: lower hm3d score #####
        
        ##### first version of distance to goal: hm3d 69？ #####
        # is_stuck_at_goal = dtg >= self.min_distance_to_goal - 0.01
        is_near_goal = dtg < self.args.NEAR_GOAL_DISTANCE or len(self.fmm_path) < 3 # old pipeline
        ##### first version of distance to goal: hm3d 69？ #####

        def reset_flags():
            self.reach_goal_count = 0
            self.min_distance_to_goal = 10000.
            self.dtg_history = [10000.]
            self.stuck_at_goal_count = 0
            self.reach_goal_count = 0

        if not is_stuck_at_goal:
            self.stuck_at_goal_count = 0
            self.min_distance_to_goal = dtg_obstacle
        else:
            self.stuck_at_goal_count += 1
                
        if is_near_goal:
            self.reach_goal_count += 1
        else:
            self.reach_goal_count = 0

        if not is_same_goal:
            reset_flags()
        
        # TODO: how to consider switch route 
        if self.found_goal and not ((self.another_floor or self.early_stair_climb) and (self.upstair_flag or self.downstair_flag) and self.start_climbing) and not self.start_climbing:
            # found_goal when you are not climbing the stair
            if self.reach_goal_count > 2: # only stop if I get close to goal
                action = 0
            if self.stuck_at_goal_count > 10 and self.dtg_metric < 1: # 0.5m
                action = 0
        else:
            # stuck criterion
            if self.stuck_at_goal_count > 10 or self.replan_count > 15:
                if self.global_state=='imagine_nav' or self.global_state=='frontier_nav' or self.global_state=='keyboard':
                    # TODO: only invalidate the frontier also if we are near the frontier?
                    self.obstacle_map[frontier_edge_map == self.best_frontier_id+1] = 1
                else:
                    self.obstacle_map[self.goal_map==1] = 1
                if self.global_state=='reperception':
                    self.reperception_object['reperception_visited'] = True
                elif self.global_state=='best_object':
                    self.best_object['visited'] = True
                reset_flags()
        
        # if len(self.dtg_history) >= 10:
        #     self.dtg_history.pop(0)
        # self.dtg_history.append(dtg_obstacle)

        # if we run out of time, stop
        if self.l_step == 499:
            action = 0

        self.is_same_goal = is_same_goal
        self.dtg = dtg
        self.dtg_obstacle = dtg_obstacle
        self.distance_to_goal_map = planner.fmm_dist

        dd_map_time = time.time()
        time_step_info += '2d map building time:%.3fs\n'%(dd_map_time - f_map_time)

        if len(self.objects) > 0:
            time_step_info += 'max similarity: %3fs\n'%(np.max(similarities))
        
        ##################################################
        ##### Checking if we arrive at another floor #####
        ##################################################
        if (self.another_floor or self.early_stair_climb) and (self.upstair_flag or self.downstair_flag) and self.start_climbing and not self.args.no_stair_climbing:
            # stair climbing termintation condition
            new_floor_pcd = self.detect_plane_points_cell(full_scene_pcd, camera_position)
            # print("Number of new floor points:", len(new_floor_pcd.points))
            
            # if we are high enough and get enough points, we are probably on another floor
            # floor hight must be higher than camera height
            self.height_diff = camera_position[1] - self.current_floor_camera_height
            # TODO: test heuristc detection spawn on stiar 
            if self.l_step < 40 and not self.args.no_deal_stair_spawn:
                # heuristically handle case when robot spawns on the stair
                self.floor_height_threshold = 0
            height_criteria = self.height_diff > self.floor_height_threshold if self.upstair_flag else self.height_diff < -self.floor_height_threshold

            new_floor_threshold = self.args.FLOOR_POINT_THRESHOLD
                
            if len(new_floor_pcd.points) > new_floor_threshold and height_criteria:
                    ground_pcd = self.plane_segmentation_xy(new_floor_pcd)
                    # print(len(ground_pcd.points)) 
                    
                    if len(ground_pcd.points) > new_floor_threshold:
                        if self.upstair_flag:
                            self.switch_upstair_count += 1
                        elif self.downstair_flag:
                            self.switch_downstair_count += 1
                        
                        self.global_state = "new_floor"
                        # self.no_frontiers_count = 0
                        # self.current_floor_camera_height = camera_position[1]
                        
                        self.reset_stair_climb_params(camera_position[1])
                        self.reset_map(camera_position)
                        if self.args.imagine_nav_planner:
                            self.imagine_nav_planner.reset() # reset scene graph in the memory
                        
                        objects_to_keep = []
                        for obj in self.objects:
                            if abs(obj['bbox'].get_center()[1] - camera_position[1])  < 1.0 or 'stairs' in obj['class_name']:
                                objects_to_keep.append(obj)
                        self.objects = MapObjectList(objects_to_keep)

        # ------------------------------------------------------------------
        ##### Update States and Visualization
        # ------------------------------------------------------------------

        self.step_times.append(time.time() - clip_s_time)

        vis_image = None
        if self.args.print_images or self.args.visualize:
            self.annotated_image  = vis_result_fast(image, detections, self.classes)
            goal_map = self.goal_map
            if (self.another_floor or self.early_stair_climb) and (self.upstair_flag or self.downstair_flag) and not self.args.no_stair_climbing:
                if np.sum(self.stair_goal_map) > 0:
                    goal_map = self.stair_goal_map
            vis_image = self._visualize(self.obstacle_map, self.explored_map, frontier_edge_map, goal_map, self.text_queries)
            # if self.args.debug:
            #     depth_dir = f'{self.dump_dir}episodes/{self.episode_label}/depth'
            #     pose_dir = f'{self.dump_dir}episodes/{self.episode_label}/pose'
            #     os.makedirs(depth_dir, exist_ok=True)
            #     os.makedirs(pose_dir, exist_ok=True)
            #     np.save(os.path.join(depth_dir, f'depth_{self.l_step:03}.npy'), observations['depth'])
            #     np.save(os.path.join(pose_dir, f'pose_{self.l_step:03}.npy'), agent_state.sensor_states["depth"].position)

        if self.args.visualize:
            if cfg.filter_interval > 0 and (self.l_step+1) % cfg.filter_interval == 0:
                self.point_sum = self.point_sum.voxel_down_sample(0.05)
            if gui_viz_queue is not None:
                serializable_obj = self.objects.to_serializable()
                gui_viz_queue.put([
                    image_rgb,
                    depth,
                    self.annotated_image,
                    serializable_obj,
                    np.asarray(self.point_sum.points),
                    np.asarray(self.point_sum.colors),
                    self.Open3D_traj,
                    self.episode_n,
                    self.episode_label,
                    plan_path,
                    transform_rgb_bgr(vis_image),
                    Open3d_goal_pose,
                    time_step_info, 
                    candidate_id
                ])
        
        ##### log perception benchmakr #####
        if self.args.save_perception_results:
            vln_logger.set_perception_benchmark_dir(f'{self.dump_dir}perception_benchmark/{self.episode_label}')
            vln_logger.save_perception_benchmark()
        # transfer_time = time.time()
        # time_step_info += 'transfer data time:%.3fs\n'%(transfer_time - dd_map_time)
        # cv2.imshow("episode_n {}".format(self.episode_label), self.annotated_image)
        # cv2.waitKey(1)
        # print(time_step_info)
        
        if self.args.save_step_data:
            step_data = dict()
            step_data['scene_id'] = self.scene_id
            step_data['episode_id'] = self.episode_id
            step_data['episode_label'] = self.episode_label
            step_data['process_label'] = self.args.process_label
            step_data['step'] = self.l_step
            step_data['global_state'] = self.global_state
            step_data['camera_position'] = camera_matrix_T
            step_data['current_grid_pose'] = self.current_grid_pose
            step_data['len_objects'] = len(self.objects)
            step_data['len_candidate_objects'] = len(candidate_objects)
            step_data['len_reperception_objects'] = len(reperception_objects)
            step_data['len_frontier_candidate_list'] = len(frontier_candidate_list)
            step_data['timestamp'] = datetime.datetime.now()
            step_data['step_time'] = self.step_times[-1]
            step_data['step_times'] = self.step_times
            step_data['action'] = action
            step_data['current_grid_pose'] = self.current_grid_pose
            step_data['cate_object'] = self.cate_object
            step_data['color_image'] = image_rgb
            step_data['annotated_image'] = self.annotated_image
            # step_data['depth_image'] = depth
            # if self.args.gt_perception:
            step_data['goal_poses_metric'] = self.goal_poses_metric

            if self.l_step % 5 == 0:
                ### save expensive data every N steps ###
                ### required by  ### 
                step_data['origins_grid'] = self.origins_grid
                step_data['occupancy_map'] = self.occupancy_map
                step_data['traversible_map'] = self.traversible_map
                step_data['global_bev_rgb_map'] = self.global_bev_rgb_map
                step_data['height_map'] = self.height_map
                step_data['gradient_map'] = self.gradient_map
                llm_results = self.imagine_nav_planner.scene_graph.get_llm_results()
                if llm_results is not None:
                    step_data['llm_results'] = llm_results
                ### other images and maps ### 
                # step_data['semantic_map'] = self.semantic_map
                # step_data['similarity_map'] = similarity_map
                step_data['object_nodes'] = self.imagine_nav_planner.scene_graph.object_nodes
                if self.args.gt_scenegraph:
                    step_data['gt_scenegraph'] = observations['gt_scenegraph']
                if hasattr(self.imagine_nav_planner, 'raw_scene_graph'):
                    step_data['raw_scene_graph'] = self.imagine_nav_planner.raw_scene_graph
                if hasattr(self.imagine_nav_planner, 'global_scene_graph'):
                    step_data['global_scene_graph'] = self.imagine_nav_planner.global_scene_graph
                if hasattr(self.imagine_nav_planner, 'predicted_global_scene_graph'):
                    step_data['predicted_global_scene_graph'] = self.imagine_nav_planner.predicted_global_scene_graph
            if action==0 or self.l_step == 499:
                # last step
                step_data['final_step'] = True
                step_data['object_list'] = self.imagine_nav_planner.scene_graph.object_list.to_serializable()

            if self.frontier_scores is not None:
                step_data['frontier_scores'] = self.frontier_scores
                step_data['best_frontier_id'] = self.best_frontier_id
                step_data['frontier_candidate_list'] = self.frontier_candidate_list

            while step_data_queue.qsize() > 10:
                # wait for the queue to be empty
                time.sleep(0.1)
            step_data_queue.put(step_data)
        else:
            print(f"[agent] [{self.args.process_label}] [{self.episode_label}] step: {self.l_step}, action: {action}, step_time: {self.step_times[-1]}")

        if self.args.debug and False:
            if self.switch_upstair_count > 0 or self.switch_downstair_count > 0:
                action = 0
    
        self.l_step += 1
        self.last_action = action
        self.previous_goal_map = self.goal_map.copy()
        self.prev_goal_candidate = self.curr_goal_candidate
        
        return action
    
    def search_navigable_path(self, original_point, offset = 0.1):
        plan_path = self.follower.get_path_points(
            original_point
        )
        
        if len(plan_path) > 1:
            return plan_path
  
        # Possible changes to each coordinate
        deltas = [-offset, offset]

        # Generate surrounding points using nested loops
        for dx in deltas:
            for dy in deltas:
                for dz in deltas:
                    new_point = (original_point[0] + dx, original_point[1] + dy, original_point[2] + dz)
                    plan_path = self.follower.get_path_points(
                        new_point
                    )
                    if len(plan_path) > 1:
                        self.habitat_goal_pose = new_point
                        
                        return plan_path
              
        return plan_path  

    def keyboard_act(self, agent_action):
        """Control the agent by keyboard"""

        control_mapping = {
            "w": (HabitatSimActions.move_forward, "FORWARD"),
            "a": (HabitatSimActions.turn_left, "LEFT"),
            "d": (HabitatSimActions.turn_right, "RIGHT"),
            "q": (HabitatSimActions.look_up, "UP"),
            "e": (HabitatSimActions.look_down, "DOWN"),
            "f": (HabitatSimActions.stop, "STOP"),
            "p": (-1, "AGENT ACTION"),
        }
        # keystroke = chr(cv2.waitKey(0))
        # keystroke = keyboard.read_key()
        action = None
        while action is None:
            keystroke = input("Enter a key: ")
            print(keystroke)
            if keystroke in control_mapping:
                action, action_name = control_mapping[keystroke]
                print(f"action: {action_name}")
            else:
                print("INVALID KEY")
        if action==HabitatSimActions.look_up:
            self.eve_angle += 30
        elif action==HabitatSimActions.look_down:
            self.eve_angle -= 30
        elif action==-1:
            action = agent_action
        # self.l_step += 1
        
        # if self.args.debug:
        #     return 0 
        
        return action

    
    def greedy_follower_act(self, plan_path):
        

        if not self.is_running:
            return None   

        action = self.follower.get_next_action(
            self.habitat_goal_pose
        )

        if not self.found_goal and action == 0:
            self.reach_goal_count += 1
            action = 2
        else:
            self.reach_goal_count = 0
        
        distance = np.linalg.norm(plan_path[0] - plan_path[1])
        high_diff = plan_path[1][1] - plan_path[0][1]
        angle_goal = math.degrees(math.asin(high_diff/distance))

        angle_agent = (360 - self.relative_angle) % 360.0
        eve_start_x = int(5 * math.sin(math.radians(angle_agent)) + self.current_grid_pose[0])
        eve_start_y = int(5 * math.cos(math.radians(angle_agent)) + self.current_grid_pose[1])
        eve_start_x = min(max(0, eve_start_x), self.map_size - 1)
        eve_start_y = min(max(0, eve_start_y), self.map_size - 1)
        
        
        if not self.use_keyboard:
            if self.another_floor and self.downstair_flag:
                if self.eve_angle > -20:
                    action = 5
                    self.eve_angle -= 30
                elif self.eve_angle < -40:
                    action = 4
                    self.eve_angle += 30
            else:
                if (self.explored_map[eve_start_x, eve_start_y] == 0 or (angle_goal - self.eve_angle) < -self.args.turn_angle/2 ) and self.eve_angle > -90:
                    action = 5
                    self.eve_angle -= 30
                elif self.explored_map[eve_start_x, eve_start_y] == 1 and (angle_goal - self.eve_angle) > self.args.turn_angle/2 and self.eve_angle < 0:
                    action = 4
                    self.eve_angle += 30
            

        # print('acton: %.3fs'%(action_e_time - action_s_time)) 
        self.l_step += 1        
        return action
    
    def fmm_act(self):
        if not self.is_running:
            return None
        
        # # heuristic greedy action, look down to see the map, and turn around to see more
        # if self.l_step < 3:
        #     action = HabitatSimActions.LOOK_DOWN
        # elif self.l_step < 6:
        #     action = HabitatSimActions.LOOK_UP
        # elif self.l_step < 6 + 12:
        #     action = HabitatSimActions.TURN_RIGHT
        # fmm planner
        # TODO: this look up and down logic is a bit buggy, it will cause the agent to do it repetetively 
        (stg_x, stg_y) = self.stg
        angle_st_goal = math.degrees(math.atan2(stg_x - self.current_grid_pose[0],
                                                stg_y - self.current_grid_pose[1]))
        angle_agent = (360 - self.relative_angle) % 360.0
        if angle_agent > 180:
            angle_agent -= 360
        # angle_agent = 360 - self.relative_angle
        relative_angle = angle_agent - angle_st_goal
        if relative_angle > 180:
            relative_angle -= 360
        if relative_angle < -180:
            relative_angle += 360

        eve_start_x = int(5 * math.sin(math.radians(self.relative_angle)) + self.current_grid_pose[0])
        eve_start_y = int(5 * math.cos(math.radians(self.relative_angle)) + self.current_grid_pose[1])
        eve_start_x = min(max(0, eve_start_x), self.map_size - 1)
        eve_start_y = min(max(0, eve_start_y), self.map_size - 1)
        # if eve_start_x >= self.map_size: eve_start_x = self.map_size-1
        # if eve_start_y >= self.map_size: eve_start_y = self.map_size-1 
        # if eve_start_x < 0: eve_start_x = 0 
        # if eve_start_y < 0: eve_start_y = 0 
        camera_holding_angle = 0
        # if found stair goal, then look down and go 
        if (self.another_floor or self.early_stair_climb) and (self.upstair_flag or self.downstair_flag) and not self.args.no_stair_climbing:
            if self.found_stair_goal and self.upstair_flag:
                camera_holding_angle = -30
            if self.downstair_flag:
                camera_holding_angle = -30
            if self.start_climbing and self.downstair_flag:
                camera_holding_angle = -60
        
        if self.explored_map[eve_start_x, eve_start_y] == 0 and self.eve_angle > -90:
            action = 5
            self.eve_angle -= 30
        elif self.explored_map[eve_start_x, eve_start_y] == 1 and self.eve_angle < camera_holding_angle:
            action = 4
            self.eve_angle += 30
        elif self.found_stair_goal and self.eve_angle > camera_holding_angle: # when climb stair keep the angel
            action = 5
            self.eve_angle -= 30
        elif self.found_stair_goal and self.eve_angle < camera_holding_angle:
            action = 4
            self.eve_angle += 30
        elif relative_angle > self.args.turn_angle:
            action = 3  # Right
        elif relative_angle < -self.args.turn_angle:
            action = 2  # Left
        # elif relative_angle > self.args.turn_angle / 2.:
        #     action = 7  # Right
        # elif relative_angle < -self.args.turn_angle / 2.:
        #     action = 6  # Left
        else:
            action = 1
        # self.l_step += 1
        # print('action: %.3fs'%(action_e_time - action_s_time)) 
        return action
    
    def merge_map(self, update_occupancy_map=False, update_traversible_map=False, update_semantic_map=False):
        """Merge maps into occupancy map and visualization map"""
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        obstacle_mask = cv2.dilate(self.obstacle_map, kernel_ellipse) == 1
        visited_mask = cv2.dilate(self.visited_vis, kernel_rect) == 1
        if self.args.temporal_collision:
            collision_mask = cv2.dilate(self.collision_map_final, kernel_rect) == 1
        else:
            collision_mask = cv2.dilate(self.collision_map, kernel_rect) == 1
        explored_mask = self.explored_map == 1
        goal_mask = cv2.dilate(self.goal_map, kernel_ellipse) == 1
        start_x, start_y = self.current_grid_pose
        start_mask = np.zeros((self.local_w, self.local_h), dtype=bool)
        start_mask[start_x - 2:start_x + 2, start_y - 2:start_y + 2] = 1
        frontier_edge_mask = self.frontier_edge_map > 0
        
        ##### old code that applys to goal map and traversible for fmm #####
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat
        ##### old code that applys to goal map and traversible for fmm #####

        if update_occupancy_map:
            # occupancy map: 0 - unexplored, 1 - obstacle, 2 - free
            occupancy_map = np.zeros((self.local_w, self.local_h))
            occupancy_map[explored_mask] = 2
            occupancy_map[obstacle_mask] = 1
            self.occupancy_map = occupancy_map

        if update_traversible_map:
            # traversible map: 0 - obstacle, 1 - free space
            # traversible_map = np.ones((self.local_w, self.local_h))
            # traversible_map[explored_mask] = 1
            traversible_map = obstacle_mask == 0
            traversible_map[visited_mask] = 1
            traversible_map[collision_mask] = 0
            traversible_map[start_mask] = 1
            self.traversible_map = traversible_map.astype(np.double)

        if update_semantic_map:
            # semantic map: 0 - unexplored, 1 - obstacle, 2 - free
            # 3 - target edge, 4 - goal, 5 - collision
            semantic_map = np.zeros((self.local_w, self.local_h))
            semantic_map[explored_mask] = 2
            semantic_map[obstacle_mask] = 1
            semantic_map[visited_mask] = 3
            semantic_map[frontier_edge_mask] = 4
            semantic_map[goal_mask] = 5
            semantic_map[collision_mask] = 6
            self.semantic_map = semantic_map

    def _get_stg(self):
        """Get short-term goal with fmm planner"""

        if (self.another_floor or self.early_stair_climb) and (self.upstair_flag or self.downstair_flag) and not self.args.no_stair_climbing:
            ###################################################################################
            ##### if going to another floor, using stiar climbing agent to overwirte goal #####
            ###################################################################################
            # if self.start_climbing:
            traversible = self.stair_traversible_map
            # else:
            #     traversible = self.traversible_map
            goal_map = np.copy(self.stair_goal_map)
            if np.sum(goal_map) == 0:
                goal_map = self.goal_map
        else:
            ################################
            ##### exploration strategy #####
            ################################
            # traversible: 0 - obstacle, 1 - free space
            traversible = self.traversible_map
            goal_map = self.goal_map
            
        traversible = np.pad(traversible, 1, constant_values=1)
        goal = np.pad(goal_map, 1, constant_values=0)
        
        planner = FMMPlanner(traversible)
        if ("plant" in self.text_queries or "tv" in self.text_queries) and \
            np.sum(goal_map) > 1:
            selem = skimage.morphology.disk(15)
        else: 
            selem = skimage.morphology.disk(5)
        goal = skimage.morphology.binary_dilation(goal, selem)
        try:
            planner.set_multi_goal(goal)
        except Exception as e:
            print('Error:', e)
            print(f'episode_label: {self.episode_label}')
            print(f'goal: {goal}')
            print(f'goal sum: {goal.sum()}')
            import traceback
            traceback.print_exc()
        self.fmm_dist = planner.fmm_dist

        path = []
        state = list(self.current_grid_pose) # copy start pos
        path.append(state)
        for i in range(10):
            stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
            state = [stg_x , stg_y]
            path.append(state)
            if stop:
                break
        if replan:
            self.replan_count += 1
        else:
            self.replan_count = 0

        # # TODO: debug test greeeday stop on fmm intead of gt 
        # if not self.found_goal and stop:
        #     self.reach_goal_count += 1
        # else:
        #     self.reach_goal_count = 0
        return path[1], stop, path

    def collision_check(self, camera_position):
        current_real_pose = [camera_position[0], camera_position[2]]
        
        if self.last_action == 1 and self.l_step > 0:
            x1, y1 = self.last_real_pose
            x2, y2 = current_real_pose
            t1 = 360-self.relative_angle
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < self.args.collision_threshold:  # Collision
                self.collision_count += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = y1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = x1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / self.args.map_resolution)+ int(self.origins_grid[0]), \
                            int(c * 100 / self.args.map_resolution)+ int(self.origins_grid[1])
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1
                        # print("collision!")
                if self.collision_count > 5: # conitnuously stuck in the same position 
                    # temporarily block a large portion of map to force the planner to switch to another route
                    temp_width = 20
                    temp_length = 4
                    for i in range(temp_length):
                        for j in range(temp_width):
                            wx = y1 + 0.05 * \
                                ((i + buf) * np.cos(np.deg2rad(t1))
                                + (j - temp_width // 2) * np.sin(np.deg2rad(t1)))
                            wy = x1 + 0.05 * \
                                ((i + buf) * np.sin(np.deg2rad(t1))
                                - (j - temp_width // 2) * np.cos(np.deg2rad(t1)))
                            r, c = wy, wx
                            r, c = int(r * 100 / self.args.map_resolution)+ int(self.origins_grid[0]), \
                                int(c * 100 / self.args.map_resolution)+ int(self.origins_grid[1])
                            [r, c] = pu.threshold_poses([r, c],
                                                        self.collision_map_temporal.shape)
                            self.collision_map_temporal[r, c] = self.l_step # record when it was blocked
                    
            else:
                self.collision_count = 0 # rest collision count
        
        # construct the final collision map
        in_effect_temporal_collision_map = (self.collision_map_temporal != 0) & (self.collision_map_temporal > self.l_step - 20) & (self.collision_map_temporal <= self.l_step)
        self.collision_map_final = np.logical_or(self.collision_map, in_effect_temporal_collision_map).astype(np.uint8)
        
        self.last_real_pose = current_real_pose   


    def _preprocess_depth(self, depth, min_d=0.5, max_d=5.0):
        # print("depth origin: ", depth.shape)
        depth = depth[:, :, 0] * 1
        # print(np.max(depth))
        # print(np.min(depth))
        # for i in range(depth.shape[1]):
        #     depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        depth = depth * max_d 

        return depth
    
    def pcd_to_grid_indx(self, point_sum):
        points = np.asarray(point_sum.points)
        
        # calculate the index for 2d map
        obs_i_values = np.floor((points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        obs_j_values = np.floor((points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        
        valid_mask = (obs_i_values >= 0) & (obs_i_values < self.local_w) & \
                    (obs_j_values >= 0) & (obs_j_values < self.local_h)
        obs_i_values = obs_i_values[valid_mask]
        obs_j_values = obs_j_values[valid_mask]
        
        return obs_i_values, obs_j_values
    
    def update_map_new(self, point_sum, camera_position, height_diff):

        explored_map = np.zeros((self.local_w, self.local_h))
        obstacle_map = np.zeros((self.local_w, self.local_h))
        
        #################################################################
        ##### select valid points within height range (same floor) ######
        #################################################################
        # print("Camera pose: ", camera_position)
        all_points = np.asarray(self.point_sum.points) 
        in_map_mask = (all_points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (all_points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (all_points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (all_points[:, 2] <= self.origins_real[1] + self.map_real_halfsize )
        all_points = all_points[in_map_mask] # in flipped image, x up, z right, y out
        height_min = camera_position[1] - 1.5 # 0.88 is habitat camera height, min value is 1.3 meters below the camera
        height_max = camera_position[1] + 0.5 # heuristically add a bit more height 
        height_mask = (all_points[:, 1] >= height_min) & (all_points[:, 1] <= height_max)
        all_points = all_points[height_mask] # filter out points that does not belong to the current floor
        
        # we treat points that are within one stair height diffrence as traversible
        mask_obstacle = (all_points[:, 1] >= (camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT + 0.3)) | \
                    (all_points[:, 1] <= (camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT - 0.3))
        points_obstacle = all_points[mask_obstacle]

        # calculate the index for 2d obstacle map
        obs_i_values = np.floor((points_obstacle[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        obs_j_values = np.floor((points_obstacle[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])

        obstacle_map[obs_i_values, obs_j_values] = 1
        self.obstacle_map = obstacle_map
        # filling small holes
        self.obstacle_map = cv2.morphologyEx(self.obstacle_map, cv2.MORPH_CLOSE, np.ones((5,5)))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        self.obstacle_map[cv2.dilate(self.visited_vis, kernel) == 1] = 0 # mark robot path as free space

        exp_i_values = np.floor((all_points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        exp_j_values = np.floor((all_points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])

        explored_map[exp_i_values, exp_j_values] = 1
        self.explored_map = explored_map
        
        # filling small holes
        self.explored_map = cv2.morphologyEx(self.explored_map, cv2.MORPH_CLOSE, np.ones((5,5)))

        # diff_ob_ex = explored_map - obstacle_map

        # if np.abs(self.eve_angle) < 10 and self.last_action != 4 and self.last_action != 5:
        #     self.obstacle_map[diff_ob_ex == 1] = 0 # free space # NOTE: multy what does this do
        
    def update_map(self, point_sum, camera_position, height_diff):

        explored_map = np.zeros((self.local_w, self.local_h))
        obstacle_map = np.zeros((self.local_w, self.local_h))

        # height range (z is down in Open3D)
        z_min = camera_position[1] - height_diff
        z_max = camera_position[1] + height_diff
        z_stair = camera_position[1] - self.args.map_height_cm / 100.0

        points = np.asarray(point_sum.points)

        mask_obstacle = (((points[:, 1] >= z_min) & (points[:, 1] <= z_max)) | \
                    (points[:, 1] <= z_stair)) & \
                    (points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (points[:, 2] <= self.origins_real[1] + self.map_real_halfsize)

        points_obstacle = points[mask_obstacle]

        mask_explored = (points[:, 1] <= z_max) & \
                    (points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (points[:, 2] <= self.origins_real[1] + self.map_real_halfsize )

        points_explored = points[mask_explored]

        # calculate the index for 2d obstacle map
        obs_i_values = np.floor((points_obstacle[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        obs_j_values = np.floor((points_obstacle[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])

        obstacle_map[obs_i_values, obs_j_values] = 1
        self.obstacle_map[obs_i_values, obs_j_values] = 1
        
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        self.obstacle_map[cv2.dilate(self.visited_vis, kernel) == 1] = 0 # mark robot path as free space

        exp_i_values = np.floor((points_explored[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        exp_j_values = np.floor((points_explored[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        
        previous_explored_map = self.explored_map.copy()

        explored_map[exp_i_values, exp_j_values] = 1
        self.explored_map[exp_i_values, exp_j_values] = 1
        
        # number of new gird been explored
        self.new_explored_area_each_step.append(np.sum(self.explored_map - previous_explored_map))

        diff_ob_ex = explored_map - obstacle_map

        if np.abs(self.eve_angle) < 10 and self.last_action != 4 and self.last_action != 5:
            self.obstacle_map[diff_ob_ex == 1] = 0 # free space
        
        return obs_i_values, obs_j_values
    
    def find_goal_on_obstacle_edge(self, nearest_point):
        '''
        Input:
            nearest_point: the point in the point cloud in meters
        '''
        def find_edge_pixel(binary_img, A, B, num_steps=1000):
            A = np.array(A)
            B = np.array(B)

            # Generate points along the line from A to B
            line_points = np.linspace(A, B, num_steps)
            
            previous_point = None
            for point in line_points:
                i, j = int(round(point[0])), int(round(point[1]))
                
                # Make sure we are within image bounds
                if i < 0 or i >= binary_img.shape[0] or j < 0 or j >= binary_img.shape[1]:
                    continue
                
                if binary_img[i, j] == 0:
                    # Return the previous point (on the 1 side)
                    return previous_point if previous_point is not None else (i, j)
                
                previous_point = (i, j)

            return None  # if not found
        
        def find_nearest_edge_pixel(binary_img, A):
            # Ensure binary image is uint8 (needed for morphology)
            binary_img = (binary_img > 0).astype(np.uint8)

            # Get edge pixels: erosion subtracts away interior, leaving edges
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(binary_img, kernel)
            edges = binary_img - eroded  # edge pixels are 1s here

            # Get coordinates of all edge pixels
            edge_coords = np.argwhere(edges == 1)

            # Compute Euclidean distances from A to each edge pixel
            A = np.array(A)
            dists = np.linalg.norm(edge_coords - A, axis=1)

            # Find nearest edge point
            nearest_idx = np.argmin(dists)
            nearest_edge = edge_coords[nearest_idx]
            return (nearest_edge[0], nearest_edge[1])  # return as (i, j)
        
        i_value = int(nearest_point[0] * 100 / self.args.map_resolution) + int(self.origins_grid[0])
        j_value = int(nearest_point[2] * 100 / self.args.map_resolution) + int(self.origins_grid[1])
        
        # find the connected component on the obstacle map of the nearest point
        obstacle_map = self.obstacle_map.copy().astype(np.uint8)
        numLabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(obstacle_map, 4, cv2.CV_32S)
        goal_obstacle_region = None
        for label in range(1, numLabels+1):
            mask = labelImage == label
            if (mask[i_value-1:i_value+2, j_value-1:j_value+2]).any():
                goal_obstacle_region = mask.astype(np.uint8) # convert to uint8 binary
                break
        
        if goal_obstacle_region is None:
            return None, None
        
        nearest_edge_point = find_nearest_edge_pixel(goal_obstacle_region, (i_value, j_value))
        goal_agent_edge_point = find_edge_pixel(goal_obstacle_region, (i_value, j_value), (self.current_grid_pose[0], self.current_grid_pose[1]))
        
        if goal_agent_edge_point is None:
            return [nearest_edge_point[0]] , [nearest_edge_point[1]]
        else:
            i_values = [goal_agent_edge_point[0], nearest_edge_point[0]]
            j_values = [goal_agent_edge_point[1], nearest_edge_point[1]]
            return i_values, j_values
            
    def get_transform_matrix(self, agent_state):
        """
        transform the habitat-lab space to Open3D space (initial pose in habitat)
        habitat-lab space need to rotate camera from 
        x right
        y up
        z back
        to 
        x forward
        y up
        z right
        Returns Pose_diff, R_diff change of the agent relative to the initial timestep
        """
        camera_position = agent_state.sensor_states["depth"].position
        camera_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

        h_camera_matrix = np.eye(4)
        h_camera_matrix[:3, :3] = camera_rotation
        h_camera_matrix[:3, 3] = camera_position

        habitat_camera_self = np.eye(4)
        habitat_camera_self[:3, :3] = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
        habitat_camera_self_aj = np.eye(4)
        habitat_camera_self_aj[:3, :3] = np.array([[0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0]])
        
        R_habitat2open3d = np.eye(4)
        R_habitat2open3d[:3, :3] = self.init_sim_rotation
        R_habitat2open3d[:3, 3] = self.init_sim_position

        O_camera_matrix = habitat_camera_self_aj @ np.linalg.inv(R_habitat2open3d) @ h_camera_matrix @ habitat_camera_self


        return O_camera_matrix
    
    def find_nearest_point_cloud(self, point_cloud, target_point):
        # create KDTree
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)

        # find the nearest point around the target point
        [k, idx, _] = kdtree.search_knn_vector_3d(target_point, 1)
        nearest_point = np.asarray(point_cloud.points)[idx[0]]
        
        return nearest_point
    
    
    def object_map_building(self, point_sum):

        points = np.asarray(point_sum.points)

        mask = (points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                (points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                (points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                (points[:, 2] <= self.origins_real[1] + self.map_real_halfsize)
                
        points_filtered = points[mask]

        i_values = np.floor((points_filtered[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        j_values = np.floor((points_filtered[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        
        return i_values, j_values

     
    def detect_plane_points_cell(self, point_sum, camera_position):
        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        # mask = (points[:, 1] <= camera_position[1] + 0.5 )
        mask = (points[:, 1] <= camera_position[1] -0.7) & (points[:, 1] > camera_position[1] - 1.0)
        # if self.args.new_mapping:
        #     robot_base_height = camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT
        #     mask = (robot_base_height - 0.3 < points[:, 1]) & (points[:, 1] < robot_base_height + 0.3)

        points_filtered = points[mask]
        colors_filtered = colors[mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
        
        self.new_floor_pcd_num = len(pcd.points)

        return pcd
    

    def remove_full_points_cell(self, point_sum, camera_position):
        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        # mask = (points[:, 1] <= camera_position[1] + 0.5 )
        mask = (points[:, 1] <= camera_position[1] + 0.5 )

        points_filtered = points[mask]
        colors_filtered = colors[mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)

        return pcd
        
    def remove_diff_floor_points_cell(self, point_sum, camera_position):
        points = np.asarray(point_sum.points)
        colors = np.asarray(point_sum.colors)

        mask = (points[:, 1] <= camera_position[1]) & (points[:, 1] > camera_position[1] - 1.3)

        points_filtered = points[mask]
        colors_filtered = colors[mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)

        return pcd


    def plane_segmentation_xy(self, pcd, distance_threshold=0.02, ransac_n=10, num_iterations=100):

        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                ransac_n=ransac_n,
                                                num_iterations=num_iterations)
        inlier_cloud = pcd.select_by_index(inliers)
        pcd.select_by_index(inliers, invert=True)
        return inlier_cloud


    def get_frontier_boundaries(self, frontier_loc, frontier_sizes, map_sizes):
        loc_r, loc_c = frontier_loc
        local_w, local_h = frontier_sizes
        full_w, full_h = map_sizes

        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
 
        return [int(gx1), int(gx2), int(gy1), int(gy2)]
    
    def save_rgbd_image(self, rgb_image, depth):
        vis_image_rgb = np.ones((480, 1280, 3)).astype(np.uint8) * 255
        vis_image_rgb[0:480, 0:640] = rgb_image 
        # Normalize the depth values to the range 0-255
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply a colormap (e.g., COLORMAP_JET)
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        vis_image_rgb[0:480, 640:1280] = depth_color
        
        ep_dir = '{}episodes/{}/rgbd/'.format(
            self.dump_dir, self.episode_label)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        fn = ep_dir + 'Vis-{}.png'.format(self.l_step)
        cv2.imwrite(fn, vis_image_rgb)
    
    def draw_goal_pose(self, img, goal_gt_poses_metric):
        # filter out the goal that are not on current floor
        in_floor_mask = np.logical_and(goal_gt_poses_metric[:, 1] > self.camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT, 
                        goal_gt_poses_metric[:, 1] < self.camera_position[1] + 0.5)
        
        valid_goal_poses_metric = goal_gt_poses_metric[in_floor_mask]
        
        # flip the image
        img = np.flipud(img).copy() # need copy to make array writable for opencv
        for i, goal_pose in enumerate(valid_goal_poses_metric):
            # convert to image coordinates
            goal_img_y = np.floor((goal_pose[0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
            goal_img_x = np.floor((goal_pose[2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
            
            # if any goal point is outside the map, skip this goal
            if goal_img_x < 0 or goal_img_x > self.map_size-1 or \
                goal_img_y < 0 or goal_img_y > self.map_size-1:
                continue
            
            # draw goal pose
            img = cv2.circle(img, (goal_img_x, goal_img_y), 5, (255, 0, 0), thickness=-1)
        
        # flip the image back
        img = np.flipud(img).copy()
        return img
    
    def draw_region_bbox(self, img, gt_scenegraph):
        self.in_room_id = []
        self.in_room_name = []
        self.in_map_room_id = []
        self.in_map_room_name = []
            
        # flip the image
        img = np.flipud(img).copy() # need copy to make array writable for opencv
        
        # determine which floor the agent is in 
        floor_avg_heights = [floor['floor_avg_height'] for floor in gt_scenegraph['floors']]
        floor_id = np.argmin(np.abs(np.array(floor_avg_heights) - self.camera_position[1]))
        
        # create color map for each region
        num_regions = len(gt_scenegraph['floors'][floor_id]['regions'])
        import matplotlib as mpl
        viridis = mpl.colormaps['gist_rainbow'].resampled(num_regions)
        colors = viridis(range(num_regions))[:, :3] * 255
        colors = colors.astype(np.uint8) 
        
        for i, region in enumerate(gt_scenegraph['floors'][floor_id]['regions']):
            id = region['id']
            name = region['caption']
            bbox = region['bbox']
            center = region['center_metric']
            
            # if height_avg < self.camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT or height_avg > self.camera_position[1] + 1:
            #     continue # skip regions that are not on this floor
            
            color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            
            # draw bbox
            bbox = np.array(bbox)
            bbox_img_y = np.floor((bbox[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
            bbox_img_x = np.floor((bbox[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
            
            # if all bbox points are outside the map, skip this region
            if np.sum(bbox_img_x < 0) == 4 or np.sum(bbox_img_x > self.map_size-1) == 4 or \
                np.sum(bbox_img_y < 0) == 4 or np.sum(bbox_img_y > self.map_size-1) == 4:
                continue
            
            # clip box to fit in map
            bbox_img_x = np.clip(bbox_img_x, 0, self.map_size-1)
            bbox_img_y = np.clip(bbox_img_y, 0, self.map_size-1)
            pts = np.vstack((bbox_img_x, bbox_img_y)).T #.reshape((-1, 1, 2))

            img = cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)
            
            # draw center text
            text_layer = np.zeros_like(img)
            center_img_y = np.floor(center[0]*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
            center_img_x = np.floor(center[2]*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
            # directly use the center in grid 
            center_img_y = region['center'][0] 
            center_img_x = region['center'][1] 
            text_layer, _ = add_text(text_layer, f"({id}: {name})", (center_img_x, self.map_size - center_img_y), font_scale=0.35, color=(120,120,120), thickness=1, horizontal_align='center', vertical_align='center')
            text_layer = np.flipud(text_layer).copy()
            mask = np.any(text_layer > 0, axis=2)
            for c in range(3):
                img[:,:,c][mask] = text_layer[:,:,c][mask]

            # determine if robot is in this region
            from matplotlib.path import Path
            polygon_path = Path(pts)
            pose_img_y = self.current_grid_pose[0]
            pose_img_x = self.current_grid_pose[1]
            contain_2d = polygon_path.contains_point((pose_img_x, pose_img_y))
            if contain_2d:
                self.in_room_id.append(id)
                self.in_room_name.append(name)
            
            self.in_map_room_id.append(id)
            self.in_map_room_name.append(name)
            
            # draw center to debug 
            # img = cv2.circle(img, (center_img_x, center_img_y), 3, (0, 0, 255))     
            # for x, y in zip(bbox_img_x, bbox_img_y):
            #     img = cv2.circle(img, (x, y), 3, (0, 0, 255))
            
            # draw all the object centers to debug
            # if self.args.debug:
            for object in region['objects']:
                img = cv2.circle(img, (object['center'][1], object['center'][0]), 3, (0, 0, 255))     
        
        img = np.flipud(img).copy()

        return img
            
    def save_similarity_map(self, map):
        depth_normalized = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply a colormap (e.g., COLORMAP_JET)
        depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        ep_dir = '{}episodes/{}/similarity/'.format(
            self.dump_dir, self.episode_label)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        fn = ep_dir + 'Vis-simi-{}.png'.format(self.l_step)
        cv2.imwrite(fn, depth_color)
    
    def get_semantic_map_vis(self, goal_map):
        sem_map = self.semantic_map.copy()
        # draw larger dot
        selem = skimage.morphology.disk(4)
        goal_mat = skimage.morphology.binary_dilation(
            goal_map, selem).astype(bool)

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 7
        
        # drawing circle around the goal
        if np.sum(goal_map)>=1:
            goal_pos = np.argwhere(goal_map == 1)[0]
            goal_fmb = skimage.draw.circle_perimeter(goal_pos[0], goal_pos[1], int(self.map_size/12-1))
            goal_fmb[0][goal_fmb[0] > self.map_size-1] = self.map_size-1
            goal_fmb[1][goal_fmb[1] > self.map_size-1] = self.map_size-1
            goal_fmb[0][goal_fmb[0] < 0] = 0
            goal_fmb[1][goal_fmb[1] < 0] = 0
            # goal_fmb[goal_fmb < 0] =0
            goal_mask = np.zeros((self.map_size, self.map_size), dtype=bool)
            goal_mask[goal_fmb[0], goal_fmb[1]] = 1
            sem_map[goal_mask] = 7
        # sem_map[:10,:20] = 1 # use this to identify the origin and the x, y axis
        
        # draw planned path 
        planned_path_mask = np.zeros((self.map_size, self.map_size))
        for waypoint in self.fmm_path:
            planned_path_mask[waypoint[0], waypoint[1]] = 1
        selem = skimage.morphology.disk(2)
        planned_path_mask = skimage.morphology.binary_dilation(
            planned_path_mask, selem).astype(bool)
        planned_path_mask = planned_path_mask == 1
        sem_map[planned_path_mask] = 9
        
        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8).tolist()) # original map, x down, z right 
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis).copy() # flip upsdie down x up, z right 
        sem_map_vis = self.draw_goal_pose(sem_map_vis, self.goal_poses_metric)
        
        if hasattr(self.imagine_nav_planner.scene_graph, 'scene_graph') and self.imagine_nav_planner.scene_graph.scene_graph is not None:
            obs_sg = self.imagine_nav_planner.scene_graph.scene_graph
            regions = [region for room in obs_sg['rooms'] for region in room['regions']]
            plot_region(sem_map_vis, regions, self.map_size, min_obj_num=3, plot_objects=False)

        # if self.args.gt_scenegraph:
        #     sem_map_vis = self.draw_region_bbox(sem_map_vis, self.gt_scenegraph)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        
        def get_contour_points(pos, origin, size=20):
            x, y, o = pos
            pt1 = (int(x) + origin[0],
                int(y) + origin[1])
            pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
                int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
            pt3 = (int(x + size * np.cos(o)) + origin[0],
                int(y + size * np.sin(o)) + origin[1])
            pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
                int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])
            return np.array([pt1, pt2, pt3, pt4])

        # draw arrow at the agent position
        pos = [self.last_grid_pose[1], int(self.map_size - self.last_grid_pose[0]), np.deg2rad(self.relative_angle)]
        agent_arrow = get_contour_points(pos, origin=(0, 0), size=10)
        color_index = 8
        color = [int(x*255) for x in reversed(color_palette[color_index*3:color_index*3+3])] # RGB to BGR for cv2
        cv2.drawContours(sem_map_vis, [agent_arrow], 0, color, -1)
        
        # draw frontier score
        if self.global_state=='imagine_nav' and self.args.debug_frontier_score:
            frontier_locs = self.frontier_candidate_list
            best_frontier_id = self.best_frontier_id
            for i, loc in enumerate(frontier_locs):
                if i==best_frontier_id:
                    sem_map_vis = cv2.circle(sem_map_vis, (loc[1], self.map_size - loc[0]), 7, (47,153,26), thickness=-1)
                    sem_map_vis, _ = add_text(sem_map_vis, f"{i}", (loc[1], self.map_size - loc[0]), font_scale=0.35, color=(255,255,255), thickness=1, horizontal_align='center', vertical_align='center')
                else:
                    sem_map_vis = cv2.circle(sem_map_vis, (loc[1], self.map_size - loc[0]), 7, (100,100,100), thickness=-1)
                    sem_map_vis, _ = add_text(sem_map_vis, f"{i}", (loc[1], self.map_size - loc[0]), font_scale=0.35, color=(255,255,255), thickness=1, horizontal_align='center', vertical_align='center')
        
            

        return sem_map_vis

    def _visualize(self, map_pred, exp_pred, map_edge, goal_map, text_queries):
        # prepare data
        task_text = f'Episode: {self.episode_label}, Instruction: "{text_queries}", Step: {self.l_step}, Action: {self.action_keys[self.last_action]}, Time: {self.step_times[-1]:.1f}s'
        state_text = f'[State] {self.state_dict[self.global_state][0]}'
        state_args = {"font_scale": 1.0, "color": self.state_dict[self.global_state][1], "thickness": 2}
        sem_map_vis = self.get_semantic_map_vis(goal_map)
        travisable_img = np.flipud(np.stack([self.traversible_map.copy()]*3, axis=-1).astype(np.uint8) * 255)

        visual_bev = self.global_bev_rgb_map.copy()
        # if self.args.gt_perception:
        visual_bev = self.draw_goal_pose(visual_bev, self.goal_poses_metric)
        if self.args.gt_scenegraph:
            # draw region bbox on rgb bev img 
            visual_bev = self.draw_region_bbox(visual_bev, self.gt_scenegraph)

        function_call_list = {
            'gpt': ['query_llm', 'llm_name', 'gpt'],
            'llama': ['query_llm', 'llm_name', 'llama'],
            'qwen': ['query_llm', 'llm_name', 'qwen'],
            'other': ['query_llm', 'llm_name', 'qwen'],
            'pred_sg': ['predict_scenegraph'],
            'pred_region': ['get_group_caption'], # generate_region_caption
        }
        function_call_count = {}
        function_call_increment = {}
        for func_label, arg_list in function_call_list.items():
            function_call_count[func_label] = vln_logger.get_function_call_count(*arg_list)
            if hasattr(self, '_last_function_call_count'):
                function_call_increment[func_label] = function_call_count[func_label] - self._last_function_call_count[func_label]
            else:
                function_call_increment[func_label] = 0
        self._last_function_call_count = function_call_count
        function_call_text = []
        for func_label, count in function_call_count.items():
            increment = function_call_increment[func_label]
            function_call_text.append(f"{func_label}: {count}[{increment}]")
        function_call_text = ', '.join(function_call_text)
        function_call_text = f"[Function Call] {function_call_text}"
        # stop criterion
        stop_criterion_text = f'[Stop/Stuck] gt_dtg={self.metrics["distance_to_goal"]:.2f}'
        # fmm_distance_image = np.ones((self.local_w, self.local_h, 3))*255
        if hasattr(self, 'dtg'):
            stop_criterion_text += f', dtg_goal={self.dtg:.2f}, dtg_obstacle={self.dtg_obstacle:.2f}, min_dtg={self.min_distance_to_goal:.2f}, dtg_metric={self.dtg_metric:.2f} stuck={self.stuck_at_goal_count}, reach={self.reach_goal_count}, same_goal={self.is_same_goal}, collision_count={self.collision_count}'
            # fig, ax = plt.subplots(figsize=(4.8, 4.8))
            # im = ax.imshow(self.distance_to_goal_map, cmap='plasma', origin='lower')
            # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # fig.canvas.draw()
            # buf = fig.canvas.tostring_rgb()
            # ncols, nrows = fig.canvas.get_width_height()
            # fmm_distance_image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            # fmm_distance_image = cv2.cvtColor(fmm_distance_image, cv2.COLOR_RGB2BGR)
            # fmm_distance_image = remove_image_border(fmm_distance_image)
        # frontier score
        frontier_score_text = []
        if self.global_state=='imagine_nav' and self.args.debug_frontier_score and self.frontier_scores is not None:
            frontier_locs = self.frontier_candidate_list
            best_frontier_id = self.best_frontier_id
            factor_score = [self.frontier_scores[key] for key in ['final', 'distance', 'exploration', 'exploitation', 'hierarchical']]
            current_line = []
            for i, loc in enumerate(frontier_locs):
                frontier_index = f"[[{i}]]" if i==best_frontier_id else f"{i}"
                if isinstance(factor_score[4]['region_name'][i], dict):
                    region_name = ', '.join([f'{caption}:{conf:.2f}' for caption, conf in factor_score[4]['region_name'][i].items()])
                else:
                    region_name = factor_score[4]['region_name'][i]
                if isinstance(factor_score[4]['pred_region'][i], dict):
                    pred_region_name = ', '.join([f'{caption}' for caption, conf in factor_score[4]['pred_region_name'][i].items()])
                else:
                    pred_region_name = factor_score[4]['pred_region_name'][i]
                if pred_region_name != '':
                    pred_region_name = f'-{pred_region_name}'
                current_line += [f"{frontier_index}: (final={factor_score[0][i]:.3f}, dist={factor_score[1][i]:.3f}, expra={factor_score[2][i]:.3f}, expit={factor_score[3][i]:.3f}, obj={factor_score[4]['object'][i]:.3f}-{factor_score[4]['object_name'][i]}, region={factor_score[4]['region'][i]:.3f}-{region_name}, pred={factor_score[4]['pred_region'][i]:.3f}{pred_region_name})"]
                # if i%2==1 or i==len(frontier_locs)-1:
                frontier_score_text += [", ".join(current_line)]
                current_line = []
        if len(frontier_score_text)>0:
            frontier_score_text = ["[Frontier Score]"] + frontier_score_text
            frontier_score_text = [{
                "type": "text",
                "args": {"font_scale": 0.85, "color": (20, 20, 20), "thickness": 2},
                "text": x
            } for x in frontier_score_text]
        # region reasoning
        reasoning_list = self.imagine_nav_planner.scene_graph.get_reasoning()
        reasoning_text = [{
            "type": "text",
            "args": {"font_scale": 0.85, "color": (20, 20, 20), "thickness": 2},
            "text": x
        } for x in reasoning_list]


        # other text
        correlation_text = f"[Correlation] obj={self.imagine_nav_planner.object_correlation}, region={self.imagine_nav_planner.region_correlation}, prediction={self.imagine_nav_planner.with_prediction}, caption={self.imagine_nav_planner.caption_method}, fusion={self.imagine_nav_planner.mixing_mode}, exploration={self.imagine_nav_planner.exploration_mode}, mixing={self.imagine_nav_planner.mixing_mode}, exploration={self.imagine_nav_planner.with_exploration:.2f}, exploitation={self.imagine_nav_planner.with_exploitation:.2f}, distance={self.imagine_nav_planner.with_distance:.2f}, llm={self.imagine_nav_planner.scene_graph_prediction_llm}"
        stair_text = f"[Multi-floor] no_frontier_count={self.no_frontiers_count}, found_goal={self.found_goal}, stair_goal={self.found_stair_goal}, another_floor={self.another_floor}, upstair={self.upstair_flag}, downstair={self.downstair_flag} "
        stair_text_2 = f"height_diff={self.height_diff:.2f}, floor_points={self.new_floor_pcd_num}, eve_angle={self.eve_angle}, n_upstair={len(self.upstair_candidate_objects)}, n_downstair={len(self.downstair_candidate_objects)}, switch_upstair={self.switch_upstair_count}, switch_downstair={self.switch_downstair_count}"
        stuck_text = f"[Flags] replan={self.replan_count}, frontier_fmm_cost={self.frontier_fmm_cost}, fmm_path_len={len(self.fmm_path)}"
        early_climb_text = f"[Early Climb] early_stair_climb={self.early_stair_climb}, explored_last_T={self.new_explored_area_in_last_T}, exploration_scores={', '.join([f'{score:.2f}' for score in self.exploration_scores])}"
        if self.args.gt_scenegraph:
            gt_sg_text = f"[GT Scene Graph] Robot in {self.in_room_name}, id: {self.in_room_id}"
            # gt_sg_text2 = f"[GT Scene Graph] {listed_regions}"
        

        # create image
        padding_w = 15
        padding_h = 25
        text_args = {"font_scale": 1.0, "color": (20, 20, 20), "thickness": 2}
        caption_args = {"font_scale": 0.7, "color": (50, 50, 50), "thickness": 2, "horizontal_align": "center"}
        vis_grid = [
            {
                "type": "text",
                "args": text_args,
                "text": task_text
            },
            {
                "type": "text",
                "args": state_args,
                "text": state_text
            },
            {
                "type": "text",
                "args": text_args,
                "text": function_call_text
            },
            {
                "type": "text",
                "args": text_args,
                "text": stop_criterion_text
            },
            {
                "type": "text",
                "args": text_args,
                "text": stuck_text
            },
            {
                "type": "text",
                "args": text_args,
                "text": correlation_text
            },
            *frontier_score_text,
            *reasoning_text,
            {
                "type": "image_row",
                "height": self.local_h,
                "image": [
                    ["annotated image", self.annotated_image],
                    ["semantic map", remove_image_border(sem_map_vis)],
                    # ["semantic map", sem_map_vis],
                    # ["semantic map", sem_map_vis],
                    # ["traversable map [stair]", remove_image_border(self.traversable_occupancy_map)],
                    ["traversable map [fmm]", remove_image_border(travisable_img)],
                    ["global bev rgb map", remove_image_border(visual_bev)]
                    # ["fmm distance map", fmm_distance_image],
                ],
            },
            {
                "type": "text",
                "args": text_args,
                "text": stair_text
            },
            {
                "type": "text",
                "args": text_args,
                "text": stair_text_2
            },
            {
                "type": "text",
                "args": text_args,
                "text": early_climb_text
            },
        ]
        
        if self.args.gt_scenegraph:
            vis_grid.append({
                "type": "text",
                "args": text_args,
                "text": gt_sg_text
            })
            # vis_grid.append({
            #     "type": "text",
            #     "args": text_args,
            #     "text": gt_sg_text2
            # })

        # if scene_graph.obs_bev_image is not None and scene_graph.pred_bev_image is not None:
        #     vis_grid.append({
        #         "type": "image_row",
        #         "height": int(self.local_h*2),
        #         "image": [
        #             ["observed graph", remove_image_border(scene_graph.obs_bev_image[..., :3][..., ::-1])],
        #             ["predicted graph", remove_image_border(scene_graph.pred_bev_image[..., :3][..., ::-1])],
        #             # ["global bev rgb map", self.global_bev_rgb_map]
        #             # ["height map", self.global_height_image],
        #             # ["height gradient", self.global_height_gradient_img],
        #             # ["fmm distance map", fmm_distance_image],
        #         ],
        #     })
        
        if (self.another_floor or self.early_stair_climb) and (self.upstair_flag or self.downstair_flag) and not self.args.no_stair_climbing:
            vis_grid.append({
                "type": "image_row",
                "height": self.local_h,
                "image": [
                    ["height map", self.global_height_image],
                    ["height gradient", self.global_height_gradient_img],
                    ["Connected traversible", self.connected_img],
                    
                ],
            })
            vis_grid.append({
                "type": "image_row",
                "height": self.local_h,
                "image": [
                    ["traversable_occupancy_map", self.traversable_occupancy_map],
                    ["stair map", self.stair_map_img],
                    ["filtered stair map", self.filtered_stair_map_img],
                ],
            })
            
        # calculate image size
        image_w = 0
        image_h = 0
        pos_x = padding_w
        pos_y = padding_h
        element_list = []
        for row in vis_grid:
            if row["type"] == "text":
                pos_x = padding_w
                row["pos"] = [pos_x, pos_y]
                textsize_w, textsize_h = add_text(None, row["text"], (0, 0), **row["args"])
                image_w = max(image_w, textsize_w)
                image_h += textsize_h + padding_h
                pos_y += textsize_h + padding_h
                element_list.append(row)
            elif row["type"] == "image_row":
                element_list.append(row)
                pos_x = padding_w
                total_width = padding_w * (len(row["image"]) - 1)
                caption_size_w, caption_size_h = add_text(None, "Caption", (0, 0), **caption_args)
                for i, (caption, img) in enumerate(row["image"]):
                    new_size_w, new_size_h = resize_with_height(img, row["height"])
                    total_width += new_size_w
                    row["image"][i].append([pos_x, pos_y])
                    row["image"][i].append([new_size_w, new_size_h])
                    caption = {"type": "text", "args": caption_args, "text": caption, "pos": []}
                    caption["pos"].append(pos_x + int(0.5*new_size_w))
                    caption["pos"].append(pos_y + new_size_h + int(0.5*padding_h))
                    element_list.append(caption)
                    pos_x += new_size_w + padding_w
                image_w = max(image_w, total_width)
                pos_y += row["height"] + int(1.5*padding_h) + caption_size_h
                image_h += row["height"] + int(1.5*padding_h) + caption_size_h
        image_w += padding_w * 2
        image_h += padding_h * 2
        # generate image
        vis_image = np.ones((image_h, image_w, 3), dtype=np.uint8) * 255
        for row in element_list:
            if row["type"] == "text":
                add_text(vis_image, row["text"], row["pos"], **row["args"])
            elif row["type"] == "image_row":
                for i, (caption, img, pos, new_size) in enumerate(row["image"]):
                    new_image = cv2.resize(img, new_size)
                    vis_image = add_img(vis_image, new_image, pos)

        # save for video
        self.vis_frames.append(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        if self.args.print_images:
            # ep_dir = f'{self.dump_dir}vis/{self.episode_label}/vis/'
            # if not os.path.exists(ep_dir):
            #     os.makedirs(ep_dir)
            # cv2.imwrite(f'{self.dump_dir}vis/{self.episode_label}/vis/vis_{self.l_step}.png', vis_image)
            # cv2.imwrite(f'{self.dump_dir}vis/{self.episode_label}/current_state.png', vis_image)
            if not os.path.exists(f'{self.dump_dir}vis'):
                os.makedirs(f'{self.dump_dir}vis')
            cv2.imwrite(f'{self.dump_dir}vis/current_state_process{self.args.process_id}.png', vis_image)
        return vis_image

    def get_global_bev_rgb_map(self, camera_position):
        all_points = np.asarray(self.point_sum.points) 
        in_map_mask = (all_points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (all_points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (all_points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (all_points[:, 2] <= self.origins_real[1] + self.map_real_halfsize )
        all_points = all_points[in_map_mask] # in flipped image, x up, z right, y out
        if self.upstair_flag:
            height_min = camera_position[1] - 1.5 # 0.88 is habitat camera height, min value is 1 meters below the camera
            height_max = camera_position[1] + 0.3 # heuristically add a bit more height 
        elif self.downstair_flag:
            height_min = camera_position[1] - 2 # 0.88 is habitat camera height, min value is 1 meters below the camera
            height_max = camera_position[1] # we need to see anything at the camera height as obstacle
        else:
            # just for visualization
            height_min = camera_position[1] - 1.5 # 0.88 is habitat camera height, min value is 1 meters below the camera
            height_max = camera_position[1] + 0.5 # heuristically add a bit more height 
        height_mask = (all_points[:, 1] >= height_min) & (all_points[:, 1] <= height_max)
        all_points = all_points[height_mask] # filter out points that does not belong to the current floor
        ###############################
        ##### global bev rgb map ######
        ###############################
        all_colors = np.asarray(self.point_sum.colors)
        all_colors = all_colors[in_map_mask]
        all_colors = all_colors[height_mask]
        global_bev_rgb_map = self.get_bev_rgb_map(all_points, all_colors)
        return global_bev_rgb_map
    
    def stair_climbing_policy(self, stair_candidate_list, camera_position):
        '''
        
        NOTE: in map img, x up, z right, y out
        NOTE: there is a move_map_and_pose function that shifts the grid map origin
            when robot is moving to the edge of the map
        '''
        #################################################################
        ##### select valid points within height range (same floor) ######
        #################################################################
        # print("Camera pose: ", camera_position)
        all_points = np.asarray(self.point_sum.points) 
        in_map_mask = (all_points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (all_points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (all_points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (all_points[:, 2] <= self.origins_real[1] + self.map_real_halfsize )
        all_points = all_points[in_map_mask] # in flipped image, x up, z right, y out
        if self.upstair_flag:
            height_min = camera_position[1] - 1.5 # 0.88 is habitat camera height, min value is 1 meters below the camera
            height_max = camera_position[1] + 0.3 # heuristically add a bit more height 
        elif self.downstair_flag:
            height_min = camera_position[1] - 2 # 0.88 is habitat camera height, min value is 1 meters below the camera
            height_max = camera_position[1] # we need to see anything at the camera height as obstacle
        else:
            # just for visualization
            height_min = camera_position[1] - 1.5 # 0.88 is habitat camera height, min value is 1 meters below the camera
            height_max = camera_position[1] + 0.5 # heuristically add a bit more height 
        height_mask = (all_points[:, 1] >= height_min) & (all_points[:, 1] <= height_max)
        all_points = all_points[height_mask] # filter out points that does not belong to the current floor
        
        ##################################################
        ##### local region mask on grid for x and y ######
        ##################################################
        # create 2m range mask
        # x_min, x_max = camera_position[0] - LOCAL_REIGION_RANGE, camera_position[0] + LOCAL_REIGION_RANGE
        # z_min, z_max = camera_position[2] - LOCAL_REIGION_RANGE, camera_position[2] + LOCAL_REIGION_RANGE
        # i_min = np.floor(x_min * 100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        # i_max = np.floor(x_max * 100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        # j_min = np.floor(z_min * 100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        # j_max = np.floor(z_max * 100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        i_min = self.current_grid_pose[0] - self.args.LOCAL_GRID_RANGE
        i_max = self.current_grid_pose[0] + self.args.LOCAL_GRID_RANGE
        j_min = self.current_grid_pose[1] - self.args.LOCAL_GRID_RANGE
        j_max = self.current_grid_pose[1] + self.args.LOCAL_GRID_RANGE
        local_map_mask = np.zeros((self.local_w, self.local_h)).astype(bool)
        local_map_mask[i_min:i_max, j_min:j_max] = True # this is a mask to indicate the local region in the grid
        local_map_mask_goal_selection = np.zeros((self.local_w, self.local_h)).astype(bool)
        local_map_mask_goal_selection[i_min+self.args.LOCAL_GRID_RANGE//2:i_max-self.args.LOCAL_GRID_RANGE//2, j_min+self.args.LOCAL_GRID_RANGE//2:j_max-self.args.LOCAL_GRID_RANGE//2] = True # this is a mask to indicate the local region in the grid specifically for selecting stiar goal
        
        ###############################
        ##### global bev rgb map ######
        ###############################
        all_colors = np.asarray(self.point_sum.colors)
        all_colors = all_colors[in_map_mask]
        all_colors = all_colors[height_mask]
        self.global_bev_rgb_map = self.get_bev_rgb_map(all_points, all_colors)
        
        ##############################################
        ##### global height and height gradient ######
        ##############################################
        global_height_map, global_gradient_map = self.create_global_height_map(all_points, height_min)
        gradient_mask = global_gradient_map < self.args.GRADIENT_THRESHOLD # need to tune
        
        ####################################################
        ##### calculate new obstacle and explored map ######
        ####################################################
        # old method
        # filling in small holes 
        kernel = np.ones((5, 5), dtype=np.uint8)
        # connected_traversable_mask = cv2.inRange(connected_traversable_mask,0.1,1)
        cv2.morphologyEx(self.explored_map.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        self.obstacle_map_multy, self.explored_map_multy = self.get_new_obstacle_explore_map(all_points, global_height_map, global_gradient_map, camera_position)
        explored_map = self.explored_map_multy
        obstacle_map = self.obstacle_map_multy
        
        ###############################################
        ##### get traversable considering stairs ######
        ###############################################
        traversable_mask = (explored_map == 1) & gradient_mask & (obstacle_map != 1) # get pixel where robot can stand
        connected_traversable_mask = self.get_connected_traversable_mask(traversable_mask) # only connected regions are traversable
        
        #############################################
        ##### semantic method to get stiar map ######
        #############################################
        
        # build a 2D map that indiates occupancy of stairs using semantics
        stair_map = self.create_stair_map(stair_candidate_list)
        # filter out invalid stairs by connectivity and gradient
        connected_traversable_stair_mask = connected_traversable_mask & stair_map.astype(bool) & (global_gradient_map > 0)
        
        ####################################
        ##### selecting goal position ######
        ####################################
        goal_ind = self.stair_goal_selection(connected_traversable_mask, connected_traversable_stair_mask, global_height_map, global_gradient_map, local_map_mask_goal_selection, camera_position)
        # print("Goal index: ", goal_ind)
        
        ##########################################
        ##### detect frontier using new map ######
        ##########################################
        target_score, target_edge_map, target_point_list = detect_frontier(explored_map, obstacle_map, self.current_grid_pose, threshold_point=8)
        
        ################################################
        ##### visualize important maps and thigns ######
        ################################################
        
        # visualize global height map and gradient
        self.global_height_image = get_pyplot_as_numpy_img(global_height_map)
        self.global_height_gradient_img = get_pyplot_as_numpy_img(global_gradient_map, cmap='hot')
        
        # visualize connected_traversable_mask
        self.connected_img = get_pyplot_as_numpy_img(connected_traversable_mask.astype(np.uint8), cmap='binary') # visualize connected traversable mask
        
        # visualize stair_mask
        self.stair_map_img = get_pyplot_as_numpy_img(stair_map.astype(np.uint8), cmap='binary') # visualize stair map
        self.filtered_stair_map_img = get_pyplot_as_numpy_img(connected_traversable_stair_mask.astype(np.uint8), cmap='binary') # visualize filtered stair map
        
        # create new map: traversable occupancy map
        sem_map = np.zeros((self.local_w, self.local_h))
        
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        if self.args.temporal_collision:
            collision_mask = self.collision_map_final == 1
        else:
            collision_mask = self.collision_map == 1
            
        visited_mask = self.visited_vis == 1
        sem_map[connected_traversable_mask] = 2 # use stiar to overwrite obstacle as free space
        sem_map[obstacle_map == 1] = 1
        sem_map[target_edge_map > 0] = 3
        sem_map[collision_mask] = 6
        sem_map[visited_mask] = 3
        
        new_obstacle_map = (sem_map == 1).astype(np.uint8)
        
        # indicating goal on goal map
        goal_map = np.zeros((self.local_w, self.local_h))
        if goal_ind is not None:
            #################################
            ##### visualize stair goal ######
            #################################
            goal_map[goal_ind[0], goal_ind[1]] = 1
            
            # selem = skimage.morphology.disk(4)
            selem = skimage.morphology.disk(1)
            goal_mat = skimage.morphology.binary_dilation(
            goal_map, selem).astype(bool)
            goal_mask = goal_mat == 1
            
            # draw larger circle to show goal
            goal_fmb = skimage.draw.circle_perimeter(goal_ind[0], goal_ind[1], int(self.map_size/16 -1))
            goal_fmb[0][goal_fmb[0] > self.map_size-1] = self.map_size-1
            goal_fmb[1][goal_fmb[1] > self.map_size-1] = self.map_size-1
            goal_fmb[0][goal_fmb[0] < 0] = 0
            goal_fmb[1][goal_fmb[1] < 0] = 0
            # goal_fmb[goal_fmb < 0] =0
            goal_mask[goal_fmb[0], goal_fmb[1]] = 1
            sem_map[goal_mask] = 4
        
        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8).tolist()) # original map, x down, z right 
        sem_map_vis = sem_map_vis.convert("RGB") 
        sem_map_vis = np.array(sem_map_vis)
        
        ##### only visual local map #####
        if self.start_climbing:
            sem_map_vis[~local_map_mask] = (0,0,0) # set non-local region to black, this is for visualization only
        
        sem_map_vis = np.flipud(sem_map_vis) # flip upsdie down x up, z right 
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        vis_image = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.traversable_occupancy_map = vis_image
        
        self.last_stair_goal_ind = goal_ind
        
        ## new_traversible map, TODO: unify to one map function
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        if self.args.temporal_collision:
            collision_mask = self.collision_map_final == 1
        else:
            collision_mask = self.collision_map == 1
        visited_mask = cv2.dilate(self.visited_vis, kernel_rect) == 1
        start_x, start_y = self.current_grid_pose
        start_mask = np.zeros((self.local_w, self.local_h), dtype=bool)
        start_mask[start_x - 2:start_x + 2, start_y - 2:start_y + 2] = 1
        
        new_traversible_map = new_obstacle_map == 0
        new_traversible_map[visited_mask] = 1
        new_traversible_map[start_mask] = 1
        new_traversible_map[collision_mask] = 0
        new_traversible_map = new_traversible_map.astype(np.double)
        
        return goal_ind, new_obstacle_map, new_traversible_map, goal_map
    
    # def generate_local_mask():
        
        
    def create_global_height_map(self, point_cloud, height_min):
        '''
        Create a height mask, starting frmo the min height
        
        Input:
            - point_cloud: numpy array of shape (N, 3) where N is the number 
                of points, and each point is (x, y, z)
            - height_min: the minimum height to consider for the height map 
                (e.g., camera height - 1.5)
        Output:
            - global_height_map: a 2D numpy array representing the height map, 
                where each cell corresponds to the height at that grid location
            - global_gradient_map: the gradient of the height map, 
                which can be used for further analysis or visualization (all 
                positive value)
        '''
        def generate_kernels():
            """
            Generate 4 different 3x3 kernels where each has one 1 and one -1 at opposite ends.

            Returns:
                np.ndarray: An array of shape (4, 3, 3) containing the four kernels.
            """
            kernels = np.zeros((4, 3, 3))  # Initialize 4 kernels with zeros
            
            positions = [((0, 0), (2, 2)),  # Top-left to bottom-right
                        ((0, 2), (2, 0)),  # Top-right to bottom-left
                        ((0, 1), (2, 1)),  # Top-center to bottom-center
                        ((1, 0), (1, 2))]  # Middle-left to middle-right

            for i, (pos1, pos2) in enumerate(positions):
                kernels[i, pos1[0], pos1[1]] = 1   # Set 1 at one end
                kernels[i, pos2[0], pos2[1]] = -1  # Set -1 at the opposite end
            
            return kernels
        # create global height map
        global_height_map = np.ones((self.local_w, self.local_h)) * (height_min)
        sorted_indices = np.argsort(point_cloud[:, 1]) # ascending order
        sorted_points = point_cloud[sorted_indices]

        mask = (sorted_points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                (sorted_points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                (sorted_points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                (sorted_points[:, 2] <= self.origins_real[1] + self.map_real_halfsize)
        sorted_points = sorted_points[mask]
        i_values = np.floor((sorted_points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        j_values = np.floor((sorted_points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        global_height_map[i_values, j_values] = sorted_points[:, 1]
        
        # filling in holes in height map
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        # global_height_map = cv2.dilate(global_height_map, kernel)
        global_height_map = cv2.morphologyEx(global_height_map, cv2.MORPH_CLOSE, np.ones((5,5)))

        # hand desgined gradient kernel that has physical unit
        kernels = generate_kernels()
        filtered_img = []
        for kernel in kernels:
            img = cv2.filter2D(src=global_height_map, ddepth=-1, kernel=kernel)
            filtered_img.append(img)
        filtered_img = np.stack(filtered_img)  # Shape: (4, height, width)
        global_gradient = np.max(np.abs(filtered_img), axis=0)  # Take the max across the 4 kernels to get the final gradient map
        
        # cv gradient 
        # global_gradient = cv2.morphologyEx(global_height_map, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
        # calculate gradient as image
        # global_gradient = np.abs(cv2.Laplacian(global_height_map_norm,cv2.CV_64F))
        
        # filling in holes in gradient map, becuase for each edge, the gradient may genreates two highlighted edges
        kernel = np.ones((3, 3), dtype=np.uint8)
        global_gradient = cv2.morphologyEx(global_gradient, cv2.MORPH_CLOSE, kernel)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        # global_gradient = cv2.dilate(global_gradient, kernel)
        
        return global_height_map, global_gradient
    
    def get_new_obstacle_explore_map(self, all_points, global_height_map, global_gradient_map, camera_position):
        explored_map = np.zeros((self.local_w, self.local_h))
        obstacle_map = np.zeros((self.local_w, self.local_h))
        
        # explored area
        explored_i_values = np.floor((all_points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        explored_j_values = np.floor((all_points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
        explored_map[explored_i_values, explored_j_values] = 1
        # remove small holes in side the region 
        kernel = np.ones((5, 5), dtype=np.uint8)
        explored_map = cv2.morphologyEx(explored_map.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # explored_map = cv2.dilate(explored_map.astype(np.uint8), np.ones((3,3))) # expand the explored region a little bit 
        
        # obstacle map = explored region but gradient is large, height is also large
        floor_height = camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT # assume the floor height is 1 meter below the camera
        obstacle_map = (explored_map == 1) & (global_gradient_map > self.args.GRADIENT_THRESHOLD) & (global_height_map - floor_height > self.args.OBSTACLE_HEIGHT)
        obstacle_map = obstacle_map.astype(np.uint8)
        # dilate obstacle area for safe travel 
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        # obstacle_map = cv2.dilate(obstacle_map, kernel)
        # mark robot path as free space
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        obstacle_map[cv2.dilate(self.visited_vis, kernel) == 1] = 0 
        
        return obstacle_map, explored_map
    
    def get_connected_traversable_mask(self, traversable_mask):
        # use connectivity to check if the robot can go to the free space
        connected_traversable_mask = np.zeros((self.local_w, self.local_h)).astype(bool)
        thresh = traversable_mask.astype(np.uint8)
        # use erode to break losely connected regions
        thresh_dialated = cv2.erode(thresh, np.ones((5,5), np.uint8), iterations=1)
        # print("After erode size: ", np.sum(thresh), np.sum(thresh_dialated))
        
        # find conected components
        numLabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(thresh_dialated, 8, cv2.CV_32S)
        for label in range(1, numLabels+1):
            mask = labelImage == label
            if (mask[self.current_grid_pose[0]-1:self.current_grid_pose[0]+2, self.current_grid_pose[1]-1:self.current_grid_pose[1]+2]).any():
                # the robot is in the connected component in 3x3 kernel
                connected_traversable_mask = traversable_mask & mask
                break
            # else if find not connected path, we can't determine a goal point
        
        # dilate to get back the eroded region
        connected_traversable_mask = cv2.dilate(connected_traversable_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) 
    
        return connected_traversable_mask.astype(bool)
    
    def create_stair_map(self, stair_candidate_list):
        # print("Number of stair candidates:", len(stair_candidate_list))
        stair_map = np.zeros((self.local_w, self.local_h))
        for stair_pcd in stair_candidate_list:
            stair_points = np.asarray(stair_pcd.points)
            in_map_mask_stair = (stair_points[:, 0] >= self.origins_real[0] - self.map_real_halfsize) & \
                    (stair_points[:, 0] <= self.origins_real[0] + self.map_real_halfsize) & \
                    (stair_points[:, 2] >= self.origins_real[1] - self.map_real_halfsize) & \
                    (stair_points[:, 2] <= self.origins_real[1] + self.map_real_halfsize )
            stair_points = stair_points[in_map_mask_stair]
            
            
            # calc 2D map index
            stair_i_values = np.floor((stair_points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
            stiar_j_values = np.floor((stair_points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])
            
            # mark stair occupancy 
            stair_map[stair_i_values, stiar_j_values] = 1    
            # stair_mask = stair_map.astype(bool)
        
        # expand search area
        kernel = np.ones((5, 5), dtype=np.uint8)
        stair_map = cv2.morphologyEx(stair_map.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return stair_map
    
    def stair_goal_selection(self, connected_traversable_mask, connected_traversable_stair_mask, 
                             global_height_map, global_gradient_map, local_map_mask, camera_position):
        goal_ind = None
        if connected_traversable_mask.any():
            if self.another_floor and self.last_stair_goal_ind is not None:
                # check if you arrived at the stair entrance:
                dtg = np.linalg.norm(np.array(self.last_stair_goal_ind) - np.array(self.current_grid_pose))
                if dtg < 5: # distance is smaller than 5 grid (25cm)
                    self.start_climbing = True
            
            # Generates two 2D arrays: row indices and col indices
            indices = np.indices((global_height_map.shape[0], global_height_map.shape[1]))  
            map_indices =  np.stack(indices, axis=-1)   # Stack them along the last dimension, eg. map_indices[2,5] = [2,5]
            
            if self.upstair_flag:
                height_mask = global_height_map > camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT + 0.1
            if self.downstair_flag:
                height_mask = global_height_map < camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT
            else:
                # just for visualization, similar to upstair
                height_mask = global_height_map > camera_position[1] - self.args.ACTUAL_CAMERA_HEIGHT - 0.5

            # traversible_height_map = global_height_map.copy()
            if self.start_climbing:
                valid_local_mask = local_map_mask & connected_traversable_mask & height_mask # slice into local map
                valid_indices = map_indices[valid_local_mask]
                valid_heights = global_height_map[valid_local_mask] # get the height values at valid indices
                sorted_ind = np.argsort(valid_heights) # ascending order of height values at valid indices
                
                if self.upstair_flag:
                    sorted_ind = sorted_ind[::-1] # reverse the order if going upstairs, so we can select the highest point first
                # If going down stairs, find the lowest point 
                
                self.climbing_step += 1
            else:
                ##### NOTE: method 1: using semantic to select stair location #####
                # go to the most likely stair entrance
                valid_mask = connected_traversable_stair_mask & height_mask # TODO: at this time, height must should be the same. define start_climbing_camera_height
                valid_indices = map_indices[valid_mask]
                valid_heights = global_height_map[valid_mask] # get the height values at valid indices
                sorted_ind = np.argsort(valid_heights) # ascending order of height values at valid indices
                
                if self.downstair_flag:
                    # for down stair, we want to go to the highest stair entrance point first
                    sorted_ind = sorted_ind[::-1]
                
                ##### NOTE: method 2: use pure geometry to filter out stair location #####
                # if self.upstair_flag:
                #     height_mask = (global_height_map > self.current_floor_camera_height - self.args.ACTUAL_CAMERA_HEIGHT + 0.3) & \
                #         (global_height_map < self.current_floor_camera_height + 0.3)
                # if self.downstair_flag:
                #     height_mask = (global_height_map > self.current_floor_camera_height - self.args.ACTUAL_CAMERA_HEIGHT - 0.5) & \
                #         (global_height_map < self.current_floor_camera_height - self.args.ACTUAL_CAMERA_HEIGHT + 0.2)
                # stair_area_mask = (connected_traversable_mask == True) & (0.1 < global_gradient_map) & (global_gradient_map < 0.3) & height_mask
                # valid_heights = global_height_map[stair_area_mask]
                # sorted_ind = np.argsort(valid_heights) # ascending order of height values at valid indices 
                
                # if self.upstair_flag:
                #     sorted_ind = sorted_ind[::-1] # reverse the order if going upstairs, so we can select the highest point first
                    
            # select the goal that for nearby (5,5) kernel all are traversible and in local region
            for i in sorted_ind:
                map_ind = valid_indices[i] # get the index of the valid height
                
                values = connected_traversable_mask[map_ind[0]-2:map_ind[0]+3, map_ind[1]-2:map_ind[1]+3]
                if values.all():
                    goal_ind = map_ind
                    break
                
        self.found_stair_goal = True if goal_ind is not None else False
        
        return goal_ind

    def get_bev_rgb_map(self, all_points, all_colors):
        global_rgb_map = np.ones((self.local_w, self.local_h, 3))
        
        sorted_indices = np.argsort(all_points[:, 1]) # ascending order of height
        sorted_points = all_points[sorted_indices]
        sorted_colors = all_colors[sorted_indices]
        i_values = np.floor((sorted_points[:, 0])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[0])
        j_values = np.floor((sorted_points[:, 2])*100 / self.args.map_resolution).astype(int) + int(self.origins_grid[1])

        valid_indices = (i_values >= 0) & (i_values < self.local_w) & (j_values >= 0) & (j_values < self.local_h)
    
        # Apply the valid indices to i_values, j_values, and sorted_colors
        i_values = i_values[valid_indices]
        j_values = j_values[valid_indices]
        sorted_colors = sorted_colors[valid_indices]
        
        global_rgb_map[i_values, j_values] = sorted_colors # higher point color will replace the lower ones
        
        global_rgb_map = (global_rgb_map * 255).astype(np.uint8)
        global_rgb_map = global_rgb_map[::-1] # flip the image to match the habitat coordinate system
        global_rgb_map = cv2.cvtColor(global_rgb_map, cv2.COLOR_RGB2BGR)
        return global_rgb_map
