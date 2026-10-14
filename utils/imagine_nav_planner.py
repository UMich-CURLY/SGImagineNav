import numpy as np

import copy
import skimage
from matplotlib import colors
from utils.generate_targetdist import parse_dict_scene_graph, compute_location_distribution, compute_entropy_per_grid, get_local_map
from utils.vln_logger import vln_logger
from utils.raycast import seperate_map, get_visible_unknown,get_headings,get_trajectory_exploration

from utils.fmm_planner import FMMPlanner 
from utils.scene_graph_utils import get_sorted_key_value_lists, frontier_to_unknown
from utils.mapping import get_camera_K
from utils.predict_scenegraph import PredictSceneGraph


class ImagineNavPlanner():
    def __init__(self, args, exp_config, clip_model_list):
        if args.no_llm:
            exp_config["utility"]['prediction'] = False
        self.clip_model_list = clip_model_list
        self.exp_config = exp_config
        self.args = args
        self.camera_K = get_camera_K(self.args.frame_width, self.args.frame_height, self.args.hfov)
        self.object_goal = None

        # utility
        config = exp_config["utility"]
        self.dump_dir = f"{args.dump_location}/{args.exp_name}/"
        self.with_exploitation = config["trade_off"]["exploitation"] if config["trade_off"]["exploitation"]>0 else 0.0
        self.with_exploration = config["trade_off"]["exploration"] if config["trade_off"]["exploration"]>0 else 0.0
        self.with_distance = config["trade_off"].get("distance", 0.5)
        self.hierarchical_room = config["hierarchical"]["room"]
        self.hierarchical_region = config["hierarchical"]["region"]
        self.hierarchical_object = config["hierarchical"]["object"]
        self.hierarchical_fusion = config["hierarchical"].get("fusion", "max")
        self.perception_conf = config["hierarchical"]["perception_conf"]
        self.exploitation_mode = config["exploitation"]["mode"] # graph, map
        self.exploration_mode = config["exploration"]["mode"] # graph, map, raycast
        self.object_correlation = config.get("object_correlation", "text") # text, visual, llm, cooccurrence
        self.region_correlation = config.get("region_correlation", "text") # text, visual, cooccurrence
        self.caption_method = config.get("caption_method", "clip") # llm, clip
        if not self.hierarchical_region:
            self.caption_method = "off"

        self.mixing_mode = config["trade_off"].get("mixing","sum")
        self.with_prediction = config["prediction"]
        self.with_gt = config.get("gt_scene_graph", False)
        self.raycast_range = config["exploration"].get("raycast_range",64)
        self.exploration_prior = config["exploration"].get("grid_prior",1/100000.)
        self.ray_fov = config["exploration"].get("fov",np.pi/3)
        self.step_discount = config["exploration"].get("step_discount",0.98)
        self.distance_decay = config["exploration"].get("distance_decay",None)
        self.scene_graph_prediction_llm = args.scene_graph_prediction_llm
        self.range = config.get("range", 32)
        self.bev_img = config.get("bev", True)
        self.prediction_mode = config.get("prediction_mode", 'Global') # Global, Local
        self.prediction_function = config.get("prediction_function", 'llm') # llm, cooccurrence

        # scene graph
        self.scene_graph = PredictSceneGraph(target='', clip_model_list=clip_model_list, object_correlation=self.object_correlation, region_correlation=self.region_correlation, caption_method=self.caption_method)
        self.reset()

    
    def reset(self):
        self.scene_graph.reset()
        self.objects = None
        self.raw_scene_graph = None
        self.global_scene_graph = None
        self.predicted_global_scene_graph = None

    def set_obj_goal(self, object_goal):
        self.object_goal = object_goal
        self.scene_graph.set_target(object_goal)
    
    def move_grid_origin(self, shift, axis):
        self.scene_graph.move_grid_origin(shift, axis)
    
    def set_step(self, step, episode_label):
        self.scene_graph.set_step(step)
        self.step = step
        self.episode_label = episode_label

    def update_scene_graph(self, objects, origins_grid, gradient_map, occupancy_map, image_history=None):
        self.objects = objects
        self.origins_grid = origins_grid
        if self.with_gt:
            return # skip the update if use gt
        if self.args.no_llm or not self.hierarchical_region:
            llm_name = 'no_llm'
        else:
            llm_name = self.args.group_caption_vlm
        self.scene_graph.update(objects, self.args.map_resolution, origins_grid, gradient_map, occupancy_map, image_history, llm_name=llm_name)


    import skimage
    def get_path_to(self,traversible,current_location,frontier):
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(5)
        goal_map = np.zeros_like(traversible)

        goal_map[frontier[0],frontier[1]]=1
        goal = skimage.morphology.binary_dilation(goal_map, selem)
        planner.set_multi_goal(goal)

        path = [current_location]
        state = current_location

        for i in range(20):
            stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
            state = [stg_x , stg_y]
            path.append(state)
            if stop:
                break
        return np.array(path)
    


    import skimage
    def get_path_to(self,traversible,current_location,frontier):
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(5)
        goal_map = np.zeros_like(traversible)

        goal_map[frontier[0],frontier[1]]=1
        goal = skimage.morphology.binary_dilation(goal_map, selem)
        planner.set_multi_goal(goal)

        path = [current_location]
        state = current_location

        for i in range(20):
            stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
            state = [stg_x , stg_y]
            path.append(state)
            if stop:
                break
        return np.array(path)
    
    def fbe(self, frontier_locations, current_location, traversible, occupancy_map, global_bev_rgb_map, gt_regions=None, force_prediction=False):
        """
        frontier: unknown area and free area 
        unknown area: not free and not obstacle 
        select a frontier using common sense
        """
        if len(frontier_locations) == 0:
            return None, [], None
        # print("doing fbe!!! saving samples")
        # print(current_location)
        # np.save("traversible_sample",traversible)
        # np.save("occupancy_sample",occupancy_map)
        self.traversible = traversible
        self.occupancy_map = occupancy_map
        self.global_bev_rgb_map = global_bev_rgb_map
        self.current_location = current_location
        frontier_locations = np.array(frontier_locations)
        # np.save("frontiers_sample",frontier_locations)
        # calculate exploration and exploitation scores
        if self.with_gt:
            if gt_regions is None:
                raise ValueError("gt_regions should be provided when with_gt is True")
            self.scene_graph.set_gt_scenegrah_from_region_dict(gt_regions)
        # for each frontier, calculate the distance score
        planner = FMMPlanner(traversible)
        planner.set_goal(current_location)

        fmm_dist = 1.0*planner.fmm_dist #make a copy, otherwise it gets changed, really bad
        if(self.distance_decay is not None):
            self.distance_weight = np.exp(-(fmm_dist*self.distance_decay))  #divide by 100-200 seems reasonable
        self.fmm_dist = fmm_dist

        all_scores = self.compute_objective(frontier_locations, force_prediction) #this must be AFTER fmm computation, since path mode depends on it...
        exploration_scores = all_scores['exploration']
        exploitation_scores = all_scores['exploitation']

        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        # min_distance = 0.75
        # min_mask = distances < min_distance
        # close_mask = (distances <= close_distance) & (distances >= min_distance)
        distance_score = 1-distances.copy()/20.
        # distance_score[min_mask] = distances[min_mask] / min_distance
        # distance_score[close_mask] = 1.0
        # distance_score[far_mask] = 1.0 - (distances[far_mask] - close_distance).clip(max=5)/5.
        all_scores['distance'] = distance_score

        # calculate the final score
        # TODO: add trade off here
        final_scores = self.with_exploitation * exploitation_scores
        # print(final_scores)
        # print(self.with_exploration * exploration_scores)
        if(self.mixing_mode=="sum"):
            final_scores += self.with_exploration * exploration_scores
            final_scores += self.with_distance * distance_score
        if(self.mixing_mode=="prod"):
            final_scores+= 0.05 #base score in case exploitation is 0
            final_scores*= (self.with_exploration * exploration_scores)
        if(self.mixing_mode=="threshprod"):
            final_scores= np.maximum(0.6,final_scores) #base score in case exploitation is 0
            final_scores*= (self.with_exploration * exploration_scores)
        if(self.mixing_mode=="mixA"):
            final_scores= np.maximum(0.6,final_scores) #base score in case exploitation is 0
            final_scores*= (self.with_exploration * exploration_scores)
            final_scores+= 0.5*self.with_exploitation * exploitation_scores #exploration ranges from 0ish to 0.2ish. 
        if(self.mixing_mode=="mixB"):
            final_scores= np.maximum(0.6,final_scores) #base score in case exploitation is 0
            final_scores*= (self.with_exploration * exploration_scores)
            final_scores+= 0.1*self.with_exploitation * exploitation_scores #exploration ranges from 0ish to 0.2ish. 
            final_scores+= 0.5 * self.with_exploration * exploration_scores
            final_scores+= self.with_distance * distance_score * 0.5
        if(self.mixing_mode=="regionFallback"): #fallback to exploration score if region is not good.
            # final_scores = exploitation_scores
            if np.max(all_scores['hierarchical']['region']) < 0.3 and np.max(all_scores['hierarchical']['pred_region']) < 0.3:
                final_scores = exploration_scores

        all_scores['final'] = final_scores
        best_frontier_id = np.argmax(final_scores)
        # # TEST: switch to explore when region score is low
        # if np.max(all_scores['hierarchical']['region']) < 0.3:
        #     best_frontier_id = np.argmax(exploration_scores)
        # else:
        #     best_frontier_id = np.argmax(final_scores)
        return all_scores, best_frontier_id, exploration_scores
    
    def compute_objective(self, frontier_locations, force_prediction=False):
        if force_prediction:
            print("force prediction!")
        occupancy_map, unknown_map, agent_coordinate, global_bev_rgb_map = self.update_maps()
        scene_graph = self.scene_graph
        if self.prediction_mode == 'Local':
            self.update_targetdist(copy.deepcopy(scene_graph.scene_graph), unknown_map)

        # Only make the prediction for frontiers with large unknown region around
        if self.exploration_mode == 'raycast':
            unknown_grids = self.geometric_raycast_based(frontier_locations, occupancy_map, radius=self.raycast_range)
            unknown_idx = [i for i, count in enumerate(unknown_grids) if count >= 0.3]
        else:
            unknown_grids = self.geometric_map_based(frontier_locations, unknown_map, radius=32)
            unknown_idx = [i for i, count in enumerate(unknown_grids) if count >= 200]
        if self.with_gt:
            unknown_idx = [i for i, count in enumerate(unknown_grids)]
        # vln_logger.info(f"Unknown Frontiers: {len(unknown_idx)}, {unknown_grid_amount}", extra={"module": "Utility"})
        exploitation_scores = []
        exploration_scores = []
        hierarchical_scores = {'object': [], 'region': [], 'pred_region': [], 'room': [], 'object_name': [], 'region_name': [], 'pred_region_name': [], 'room_name': []}
        zero_hierarchical_scores = {'object': 0, 'region': 0, 'pred_region': 0, 'room': 0, 'object_name': '', 'region_name': '', 'pred_region_name': '', 'room_name': ''}

        # TODO: continue to remove redundant codes
        if self.prediction_mode == 'Global':
            unknown_frontiers = frontier_locations[unknown_idx]
            global_scene_graph = copy.deepcopy(scene_graph.obs_sg)
            # global_scene_graph = scene_graph.prune_subgraph(copy.deepcopy(scene_graph.scene_graph), agent_coordinate, threshold=320)
            predicted_global_scene_graph = None
            try:
                if self.global_scene_graph is not None and self.global_scene_graph == global_scene_graph:
                    predicted_global_scene_graph = self.predicted_global_scene_graph
            except:
                pass
            if self.with_prediction and len(unknown_frontiers) > 0 and predicted_global_scene_graph is None and self.step % 10 == 0 or force_prediction:
                unknown_centers = frontier_to_unknown(frontier_locations, occupancy_map, radius=64)
                if self.prediction_function == 'llm':
                    print(f'Predicting global scene graph with llm')
                    predicted_global_scene_graph = scene_graph.predict_global_scenegraph_TopK_WithNumber(self.dump_dir, global_bev_rgb_map, global_scene_graph, unknown_centers, agent_coordinate, llm_name=self.scene_graph_prediction_llm, vlm=self.bev_img)
                elif self.prediction_function == 'cooccurrence':
                    predicted_global_scene_graph = scene_graph.predict_global_scenegraph_cooccurrence(self.dump_dir, global_bev_rgb_map, global_scene_graph, unknown_centers, agent_coordinate, llm_name=self.scene_graph_prediction_llm, vlm=self.bev_img)
            if self.args.save_scene_graph:
                scene_graph.plot_and_save(self.dump_dir, self.episode_label, self.step, self.scene_graph_prediction_llm, global_bev_rgb_map, frontier_locations, agent_coordinate, global_scene_graph, predicted_global_scene_graph)
            # dt = 0
            exploration_scores = []

            for i, frontier in enumerate(frontier_locations):
                sub_scene_graph = scene_graph.prune_subgraph(copy.deepcopy(scene_graph.scene_graph), frontier, threshold=self.range, pred_threshold=self.range*2)
                # exploitation
                if self.with_prediction and i in unknown_idx:
                    if self.with_gt:
                        predicted_scene_graph = scene_graph.get_gt_scenegrah(frontier, self.scene_graph_prediction_llm, threshold=self.range)
                    else:
                        if predicted_global_scene_graph is None:
                            predicted_scene_graph = sub_scene_graph
                        else:
                            predicted_scene_graph = scene_graph.prune_subgraph(copy.deepcopy(predicted_global_scene_graph), frontier, threshold=self.range, pred_threshold=self.range*2)
                else:
                    predicted_scene_graph = sub_scene_graph
                exploitation_score, hierarchical_score = self.exploitation(frontier, unknown_map, predicted_scene_graph) if self.with_exploitation else [0.0, zero_hierarchical_scores]
                exploitation_scores.append(exploitation_score)
                for key, value in hierarchical_score.items():
                    hierarchical_scores[key].append(value)
                # exploration
                if(self.exploration_mode=='path' or 'pathmean'):

                    # for frontier in frontier_locations:
                    path = self.get_path_to(self.traversible,self.current_location,frontier)
                    headings = get_headings(path)
                    #subsample the path if its really long
                    if(len(path)>12):
                        # indices = np.random.choice(2,len(path)-1)
                        indices = np.linspace(0, len(path)-1, num=12, dtype=int)
                        path = path[indices]
                        headings = headings[indices]
                    step_cost = self.fmm_dist[path[:,0].astype(int),path[:,1].astype(int)] #proportional to number of steps needed to reach this part of the path
                    gamma = np.array(np.power(self.step_discount,step_cost))
                    gamma/=np.max(gamma)
                    mask,_ = get_trajectory_exploration((path,headings),(self.occupancy_map==1),(self.occupancy_map==0),gamma,self.exploration_prior,self.ray_fov,20,self.raycast_range)
                    if(self.distance_decay is not None):
                        mask*=self.distance_weight

                    #decide how to score according to the path. 
                    if(self.exploration_mode=='path'):
                        total_probability = 1-np.prod(1-mask)
                    else:
                        count = 1.0 * (self.raycast_range*2) ** 2
                        total_probability = np.sum(np.count_nonzero(mask))/count/len(path)
                    exploration_scores.append(total_probability)
                    # dt+=(time.time()-t0)
                    # print("exploration:"+str(exploration_scores))       
                elif self.exploration_mode == 'raycast':
                    exploration_score = unknown_grids[i] # if self.with_exploration else 0.0
                    exploration_scores.append(exploration_score)

                else:
                    exploration_score = self.exploration(frontier, unknown_map, sub_scene_graph, sub_scene_graph) # if self.with_exploration else 0.0
                    exploration_scores.append(exploration_score)
            self.predicted_global_scene_graph = predicted_global_scene_graph
            self.global_scene_graph = global_scene_graph
            if hasattr(scene_graph, 'raw_scene_graph'):
                self.scene_graph.raw_scene_graph = scene_graph.raw_scene_graph
            else:
                self.raw_scene_graph = scene_graph.scene_graph
            exploration_scores = np.array(exploration_scores)

            # print("path exploration time spent: %f" % dt)


        elif self.prediction_mode == 'Local':
            for i, frontier in enumerate(frontier_locations):
                # prune subgraph
                # TODO: should threshold be self.range here?
                sub_scene_graph = scene_graph.prune_subgraph(copy.deepcopy(scene_graph.scene_graph), frontier, threshold=self.range, pred_threshold=self.range*2)
                if self.with_prediction and i in unknown_idx:
                    if self.with_gt:
                        predicted_scene_graph = scene_graph.get_gt_scenegrah(frontier, self.scene_graph_prediction_llm)
                    else:
                        pruned_scene_graph = scene_graph.prune_subgraph(copy.deepcopy(scene_graph.scene_graph), frontier, threshold=320, pred_threshold=320)
                        predicted_scene_graph = scene_graph.predict_highlevel_scenegraph(pruned_scene_graph, frontier, llm_name=self.scene_graph_prediction_llm)
                        if predicted_scene_graph is None:
                            predicted_scene_graph = sub_scene_graph
                    exploitation_score, hierarchical_score = self.exploitation(frontier, unknown_map, predicted_scene_graph) if self.with_exploitation else  [0.0, zero_hierarchical_scores]
                    exploration_score = self.exploration(frontier, unknown_map, sub_scene_graph, predicted_scene_graph) if self.with_exploration else 0.0
                    if self.with_gt:
                        if self.args.save_scene_graph:
                            scene_graph.plot_and_save(self.dump_dir, self.episode_label, self.step, self.scene_graph_prediction_llm, occupancy_map, frontier, i, agent_coordinate, sub_scene_graph, predicted_scene_graph, scene_graph.scene_graph, scene_graph.scene_graph_gt)
                    else:
                        if self.args.save_scene_graph:
                            scene_graph.plot_and_save(self.dump_dir, self.episode_label, self.step, self.scene_graph_prediction_llm, occupancy_map, frontier, i, agent_coordinate, sub_scene_graph, predicted_scene_graph, scene_graph.scene_graph, scene_graph.scene_graph)
                else:
                    exploitation_score = self.exploitation(frontier, unknown_map, sub_scene_graph) if self.with_exploitation else  [0.0, zero_hierarchical_scores]
                    exploration_score = self.exploration(frontier, unknown_map, sub_scene_graph, sub_scene_graph) if self.with_exploration else 0.0
                    if self.args.save_scene_graph:
                        scene_graph.plot_and_save(self.dump_dir, self.episode_label, self.step, self.scene_graph_prediction_llm, occupancy_map, frontier, agent_coordinate, sub_scene_graph, sub_scene_graph)
                exploitation_scores.append(exploitation_score)
                exploration_scores.append(exploration_score)
                for key, value in hierarchical_score.items():
                    hierarchical_scores[key].append(value)
        return {'exploitation': np.array(exploitation_scores), 'exploration': np.array(exploration_scores), 'hierarchical': hierarchical_scores}

    def update_targetdist(self, scene_graph, unknown_map):
        grid_size = unknown_map.shape[-2:]
        target_distribution = self.semantic_map_based(scene_graph, grid_size, fusion=True)
        target_distribution = target_distribution * unknown_map
        vln_logger.view_entropy_map(target_distribution, "distribution", (850, 250), reduce_size=4)

        entropy = compute_entropy_per_grid(target_distribution)
        vln_logger.view_entropy_map(entropy, "entropy", (1100, 250), reduce_size=4)

    def update_maps(self):
        agent_coordinate = self.origins_grid
        global_bev_rgb_map = copy.deepcopy(self.global_bev_rgb_map)[::-1][...,::-1]

        # occupancy map: 0 - unexplored, 1 - obstacle, 2 - free
        occupancy_map = self.occupancy_map

        # Define RGB colors
        unknown_rgb = colors.to_rgb('#FFFFFF')   # unexplored
        free_rgb = colors.to_rgb('#E7E7E7')      # free space
        obstacle_rgb = colors.to_rgb('#A2A2A2')  # obstacle

        # Create an RGB image from the occupancy map
        h, w = occupancy_map.shape
        map_rgb = np.zeros((h, w, 3), dtype=np.float32)

        # Set colors based on occupancy values
        map_rgb[occupancy_map == 0] = unknown_rgb    # unexplored
        map_rgb[occupancy_map == 1] = obstacle_rgb   # obstacle
        map_rgb[occupancy_map == 2] = free_rgb       # free

        # Optionally store or display the result
        # resolution = 5
        # agent_coordinate = (int(full_pose[0]*100/resolution), int(full_pose[1])*100/resolution)
        
        unknown_map = (copy.deepcopy(occupancy_map) == 0) * 1.0
        return map_rgb, unknown_map, agent_coordinate, global_bev_rgb_map

    
    def geometric_map_based(self, frontier_locations, unknown_map, radius=32):
        '''
        Count the unknown regions around frontiers
        '''
        unknown_grids = []
        for i, loc in enumerate(frontier_locations):
            sub_map = get_local_map(loc, unknown_map, radius)
            # Apply the mask and count non-zero elements efficiently
            unknown_grids.append(sub_map.sum())
        return unknown_grids
    
    def semantic_graph_based(self, scene_graph, fusion='mean'):
        '''
        Get the semantic correlation between target and scene graph nodes: clip feature simularity
        Options: hierarchical semantic levels: room, region, node
        Options: with/without perception confidence 
        '''
        def get_corr_score(nodes, node_type='region'):
            # Extract the spatial coordinates and category scores from the nodes
            if len(nodes) == 0:
                return None
            corr_scores = []
            for n in nodes:
                if isinstance(n['corr_score'], list):
                    # topk score, use weighted sum
                    captions, confidences = get_sorted_key_value_lists(n['caption'])
                    weighted_corr_score = 0.
                    conf_sum = 0.
                    for conf, corr_score in zip(confidences, n['corr_score']):
                        weighted_corr_score += conf * corr_score
                        conf_sum += conf
                    weighted_corr_score /= conf_sum + 1e-6
                    corr_scores.append(weighted_corr_score)
                else:
                    corr_scores.append(n['corr_score'])
            corr_scores = np.array(corr_scores)# (N,1)

            if node_type == 'object' and self.perception_conf:
                conf = np.array([n.get('confidence', 1.0) for n in nodes])# (N,1)
                corr_scores = conf.flatten() * corr_scores  # Shape: (N,)
            for node in nodes:
                if node.get('caption', '') == '':
                    continue
                # vln_logger.info(f"{node_type}: {node.get('center', [0,0])}, {node['caption']}, {node['corr_score']}", extra={"module": "Node.score"})
            return corr_scores
        
        # Placeholder for semantic graph-based method implementation
        room_nodes = scene_graph.get("rooms", [])
        region_nodes = []
        object_nodes = []
        pred_region_nodes = []
        for room in room_nodes:
            if "regions" in room:
                obs_regions = [region for region in room["regions"] if not region.get("predicted", False)]
                region_nodes.extend(obs_regions)
                for region in obs_regions:
                    object_nodes.extend(region.get("objects", []))
                pred_regions = [region for region in room["regions"] if region.get("predicted", False)]
                pred_region_nodes.extend(pred_regions)

        score = []
        hierarchical_scores = {'object': 0, 'region': 0, 'pred_region': 0, 'room': 0, 'object_name': '', 'region_name': '', 'pred_region_name': '', 'room_name': ''}
        if self.hierarchical_object:
            object_scores = get_corr_score(object_nodes, node_type='object')
            if object_scores is not None:
                score.append(object_scores.max())
                hierarchical_scores['object'] = object_scores.max()
                hierarchical_scores['object_name'] = object_nodes[object_scores.argmax()]['caption']
        if self.hierarchical_region:
            region_scores = get_corr_score(region_nodes, node_type='region')
            if region_scores is not None:
                score.append(region_scores.max())
                hierarchical_scores['region'] = region_scores.max()
                hierarchical_scores['region_name'] = region_nodes[region_scores.argmax()]['caption']
        if self.with_prediction:
            pred_region_scores = get_corr_score(pred_region_nodes, node_type='region')
            if pred_region_scores is not None:
                score.append(pred_region_scores.max())
                hierarchical_scores['pred_region'] = pred_region_scores.max()
                best_pred_region = pred_region_nodes[pred_region_scores.argmax()]
                hierarchical_scores['pred_region_name'] = f"{best_pred_region['id']}: {best_pred_region['caption']}"
        if self.hierarchical_room:
            room_scores = get_corr_score(scene_graph.get("rooms", []), node_type='room')
            if room_scores is not None:
                score.append(room_scores.max())
                hierarchical_scores['room'] = room_scores.max()
                hierarchical_scores['room_name'] = room_nodes[room_scores.argmax()]['caption']
        if fusion == 'mean':
            score = sum(score)/(len(score)+1e-6) if len(score) > 0 else 0.0
        elif fusion == 'sum':
            score = sum(score) if len(score) > 0 else 0.0
        elif fusion == 'max':
            score = max(score) if len(score) > 0 else 0.0
        return score, hierarchical_scores

        
    def semantic_map_based(self, scene_graph, grid_size, fusion=True):
        '''
        Get the target distribution across the map: oriented-around the semantic concepts (objects/region/room)
        Options: hierarchical semantic levels: room, region, node
        return: target distribution map
        '''
        # Placeholder for semantic map-based method implementation
        current_nodes = parse_dict_scene_graph(scene_graph)
        room_nodes, region_nodes, object_nodes = current_nodes
        target_distribution = []
        if self.hierarchical_object:
            obj_scores = compute_location_distribution(object_nodes, node_type='object', grid_size=grid_size, sigma=10)
            obj_prob_map = obj_scores[0] / obj_scores[1] # 0 is the score, 1 is the count number
            target_distribution.append(obj_prob_map)
            vln_logger.view_entropy_map(obj_prob_map, f"object (min: {np.min(obj_prob_map)}, max: {np.max(obj_prob_map):.3f})", save=f"dump/debug_figs/{vln_logger.experiment_name}/obj_prob.png")
        if self.hierarchical_region:
            region_scores = compute_location_distribution(region_nodes, node_type='region', grid_size=grid_size, sigma=20)
            region_prob_map = region_scores[0] / region_scores[1]
            target_distribution.append(region_prob_map)
            vln_logger.view_entropy_map(region_prob_map, f"region (min: {np.min(region_prob_map)}, max: {np.max(region_prob_map):.3f})", save=f"dump/debug_figs/{vln_logger.experiment_name}/region_prob.png")
        if self.hierarchical_room:
            room_scores = compute_location_distribution(room_nodes, node_type='room', grid_size=grid_size, sigma=50)
            room_prob_map = room_scores[0] / room_scores[1]
            target_distribution.append(room_prob_map)
            vln_logger.view_entropy_map(room_prob_map, f"room (min: {np.min(room_prob_map)}, max: {np.max(room_prob_map):.3f})", save=f"dump/debug_figs/{vln_logger.experiment_name}/room_prob.png")
        if fusion:
            # average across the three distributions
            target_distribution = np.stack(target_distribution).mean(axis=0)
        return target_distribution

    
    def information_gain_map_based(self, scene_graph, predicted_scene_graph, grid_size):
        # Placeholder for information gain map-based method implementation
        # if diff:
        #     target_distribution = self.semantic_map_based(scene_graph, fusion=False)
        #     target_distribution_diff = self.semantic_map_based(predicted_scene_graph, fusion=False)
        #     predicted_target_distribution = []
        #     for i in range(len(target_distribution)):
        #         pred_dist = (target_distribution[i][0] + target_distribution_diff[i][0]) / (target_distribution[i][1] + target_distribution_diff[i][1])
        #         predicted_target_distribution.append(pred_dist)
        #     target_distribution = np.stack(target_distribution) / (len(target_distribution)+1e-6)
        #     predicted_target_distribution = np.stack(predicted_target_distribution) / (len(predicted_target_distribution)+1e-6)
        # else:
        target_distribution = self.semantic_map_based(scene_graph, grid_size, fusion=True)
        predicted_target_distribution = self.semantic_map_based(predicted_scene_graph, grid_size, fusion=True)
        entropy = compute_entropy_per_grid(target_distribution)
        predicted_entropy = compute_entropy_per_grid(predicted_target_distribution)
        information_gain = predicted_entropy - entropy
        return information_gain
    
    def information_gain_graph_based(self):
        # Placeholder for information gain graph-based method implementation
        pass
    
    def exploration(self, frontier, unknown_map, scene_graph, predicted_scene_graph, radius=32):
        def rescale(x):
            y = 1.0/(1+np.exp(-0.1 * (x-30)))
            return y
        
        if self.exploration_mode == 'map':
            grid_size = unknown_map.shape[-2:]
            information_gain = self.information_gain_map_based(scene_graph, predicted_scene_graph, grid_size)
            unknown_information_gain = unknown_map * information_gain
            sub_map = get_local_map(frontier, unknown_information_gain, radius)
            score = rescale(sub_map.sum())
            # vln_logger.view_entropy_map(unknown_information_gain, "entropy", (1100, 250), reduce_size=4)
        return score
    
    def exploitation(self, frontier, unknown_map, scene_graph, radius=32):
        if self.exploitation_mode == 'map':
            grid_size = unknown_map.shape[-2:]
            target_dist = self.semantic_map_based(scene_graph, grid_size)
            sub_map = get_local_map(frontier, target_dist, radius)
            score = sub_map.max()
            
            target_dist_vis = copy.deepcopy(target_dist)
            target_dist_vis[~unknown_map] = 0
            # vln_logger.view_entropy_map(target_dist_vis, "distribution", (850, 250), reduce_size=4)
        elif self.exploitation_mode == 'graph':
            score, hierarchical_scores = self.semantic_graph_based(scene_graph, fusion=self.hierarchical_fusion)
        return score, hierarchical_scores

    def geometric_raycast_based(self,frontier_locations,occupancy_map, radius = 64):
        unknown_grids = []
        count = 1.0 * (radius*2) ** 2
        unknown,obstacle,free = seperate_map(occupancy_map)
        for i, loc in enumerate(frontier_locations):
            mask = get_visible_unknown(loc,600,obstacle,unknown,radius)
            unknown_grids.append(np.count_nonzero(mask)/count)

        return unknown_grids
    
    
    def observation_model_based():
        '''
        Exploration gain: the unknown region in FOV given the raycasting
        '''
        return

    def uncertainty_aware():
        return

    def compute_objective_Direct(self, frontier_locations, scene_graph, target):
        raise NotImplementedError("compute_objective_Direct is not maintained")
        scene_graph = self.update_scenegraph(scene_graph, target)
        occupancy_map, unknown_map, agent_coordinate, global_bev_rgb_map = self.update_maps()
        self.update_targetdist(copy.deepcopy(scene_graph.scene_graph), unknown_map)

        # Only make the prediction for frontiers with large unknown region around
        unknown_grids = self.geometric_map_based(frontier_locations, unknown_map, radius=32)
        unknown_idx = [i for i, count in enumerate(unknown_grids) if count >= 200]
        unknown_grid_amount = np.array(unknown_grids)[unknown_idx]
        vln_logger.info(f"Unknown Frontiers: {len(unknown_idx)}, {unknown_grid_amount}", extra={"module": "Utility"})
        exploitation_scores = []
        exploration_scores = []

        frontiers = [[frontier_locations[i][1], frontier_locations[i][0]] for i in range(len(frontier_locations))]
        global_scene_graph = copy.deepcopy(scene_graph.scene_graph)
        predicted_scores = scene_graph.predict_scores(occupancy_map, global_scene_graph, frontiers, agent_coordinate, llm_name=self.scene_graph_prediction_llm, vlm=self.bev_img)
        for i, frontier in enumerate(frontier_locations):
            if predicted_scores is None or len(predicted_scores) != len(frontier_locations):
                exploitation_score = 0.0
            else:
                exploitation_score = predicted_scores[i]
            exploration_score = 0.0
            # vln_logger.info(f"Frontier: {frontier}, Exploitation: {exploitation_score}, Exploration: {exploration_score}", extra={"module": "Utility"})

            exploitation_scores.append(exploitation_score)
            exploration_scores.append(exploration_score)
        exploitation_scores = np.array(exploitation_scores)
        exploration_scores = np.array(exploration_scores)
        scores = self.with_exploitation * exploitation_scores + self.with_exploration * exploration_scores
        frontiers = {'spatial_locs': frontier_locations, 'exploitation_scores': exploitation_scores, 'exploration_scores': exploration_scores}
        vln_logger.update_frontiers(frontier_dict=frontiers)
        return scores
    
    def compute_objective_Detailed(self, frontier_locations, scene_graph, target):
        raise NotImplementedError("compute_objective_Detailed is not maintained")
        scene_graph = self.update_scenegraph(scene_graph, target)
        occupancy_map, unknown_map, agent_coordinate, global_bev_rgb_map = self.update_maps()
        self.update_targetdist(copy.deepcopy(scene_graph.scene_graph), unknown_map)

        # Only make the prediction for frontiers with large unknown region around
        unknown_grids = self.geometric_map_based(frontier_locations, unknown_map, radius=32)
        unknown_idx = [i for i, count in enumerate(unknown_grids) if count >= 200]
        unknown_grid_amount = np.array(unknown_grids)[unknown_idx]
        vln_logger.info(f"Unknown Frontiers: {len(unknown_idx)}, {unknown_grid_amount}", extra={"module": "Utility"})
        exploitation_scores = []
        exploration_scores = []
        for i, frontier in enumerate(frontier_locations):
            # prune subgraph
            sub_scene_graph = scene_graph.prune_subgraph(copy.deepcopy(scene_graph.scene_graph), frontier, threshold=self.range)
            if self.with_prediction and i in unknown_idx:
                if self.with_gt:
                    predicted_scene_graph = scene_graph.get_gt_scenegrah(frontier, self.scene_graph_prediction_llm)
                else:
                    predicted_scene_graph = scene_graph.predict_scenegraph(sub_scene_graph, frontier, llm_name=self.scene_graph_prediction_llm)
                exploitation_score = self.exploitation(frontier, unknown_map, predicted_scene_graph) if self.with_exploitation else 0.0
                exploration_score = self.exploration(frontier, unknown_map, sub_scene_graph, predicted_scene_graph) if self.with_exploration else 0.0
                scene_graph.plot_and_save(self.scene_graph_prediction_llm, occupancy_map, frontier, i, agent_coordinate, sub_scene_graph, predicted_scene_graph, scene_graph.scene_graph, scene_graph.scene_graph_gt)
            else:
                exploitation_score = self.exploitation(frontier, unknown_map, sub_scene_graph) if self.with_exploitation else 0.0
                exploration_score = self.exploration(frontier, unknown_map, sub_scene_graph, sub_scene_graph) if self.with_exploration else 0.0
                scene_graph.plot_and_save(self.scene_graph_prediction_llm, occupancy_map, frontier, i, agent_coordinate, sub_scene_graph, sub_scene_graph, scene_graph.scene_graph, scene_graph.scene_graph)
            # vln_logger.info(f"Frontier: {frontier}, Exploitation: {exploitation_score}, Exploration: {exploration_score}", extra={"module": "Utility"})

            exploitation_scores.append(exploitation_score)
            exploration_scores.append(exploration_score)
        exploitation_scores = np.array(exploitation_scores)
        exploration_scores = np.array(exploration_scores)
        scores = self.with_exploitation * exploitation_scores + self.with_exploration * exploration_scores
        frontiers = {'spatial_locs': frontier_locations, 'exploitation_scores': exploitation_scores, 'exploration_scores': exploration_scores}
        vln_logger.update_frontiers(frontier_dict=frontiers)
        return scores