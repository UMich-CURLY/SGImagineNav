r'''
This is the global logger for vln. VLNLogger is created from the python logger, 
and makeRecord function is overridden to add extra fields to the log record.

.. code:: py

    from vln.utils.vln_logger import vln_logger
    
    vln_logger.info("This is a log message", extra={"module": "module_name"})

The above codes print, for example, the following log message:

[2025-02-14 15:00:00,000]:[VLN]:[module_name] This is a log message

Also this logger is responsible for logging visualizations. 

'''

import logging
from logging import _logRecordFactory
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json

from utils.image_process import (
    add_resized_image,
    add_text,
    draw_bbox_with_label
)

class VLNLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format_str=None,
        dateformat=None,
        style="%",
        module="global",
    ):
        super().__init__(name, level)
        self.module = module
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)  # type:ignore
        else:
            handler = logging.StreamHandler(stream)  # type:ignore
        self._formatter = logging.Formatter(format_str, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)
        self.step = 0
        self.episode = 0
        
        ### --- Visualization logging --- ###
        self.visualize_image = None
        self.visualize_history = []
        self.entropy_maps = {}
        self.sg_image = None
        self.sg_pred_image = None
        self.frontiers = {}
        self.dump_dir = None

        ### --- Function Call logging --- ###
        self.function_call_history = {}
        
        ### --- Perception benchmark data logging --- ###
        self.perception_benchmark_dir = None
        self.curr_rgb = None
        self.curr_semantic = None
        self.curr_depth = None
        self.curr_detection = None
        # self.detection_list = []
        self.pose_list = []
    
    def reset(self):
        self.visualize_image = None
        self.visualize_history = []
        self.entropy_maps = {}
        self.sg_image = None
        self.sg_pred_image = None
        self.function_call_history = {}
        
        ### --- Perception benchmark data logging --- ###
        self.perception_benchmark_dir = None
        self.curr_rgb = None
        self.curr_semantic = None
        self.curr_depth = None
        self.curr_detection = None
        # self.detection_list = []
        self.pose_list = []
    
    def set_dump_dir(self, dump_dir):
        self.dump_dir = dump_dir
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
    
    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)

    def update_frontiers(self, frontier_dict):
        self.frontiers.update(frontier_dict)

    # override the makeRecord method to add extra fields to the log record
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        """
        A factory method which can be overridden in subclasses to create
        specialized LogRecords.
        
        Giving module name, print an extra field in the log record
        """
        rv = _logRecordFactory(name, level, fn, lno, msg, args, exc_info, func,
                             sinfo)
        rv.__dict__['module'] = self.module
        if extra is not None:
            for key in extra:
                if key in ['module']:
                    pass # allow overwriting module field
                elif (key in ["message", "asctime"]) or (key in rv.__dict__):
                    raise KeyError("Attempt to overwrite %r in LogRecord" % key)
                rv.__dict__[key] = extra[key]
        return rv

    ### --- record episode data --- ###
    def set_step(self, step):
        self.step = step

    def set_episode(self, episode):
        self.episode = episode
    
    def set_scene(self, scene):
        self.scene = scene
    
    def set_sim(self, sim):
        self.sim = sim

    def set_agent(self, agent):
        self.agent = agent
    
    def set_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name
        
    ### --- record episode data --- ###
    def set_step(self, step):
        self.step = step
    
    ### --- Perception benchmark data logging --- ###
    def set_perception_benchmark_dir(self, perception_benchmark_dir):
        self.perception_benchmark_dir = perception_benchmark_dir
        if not os.path.exists(perception_benchmark_dir):
            os.makedirs(perception_benchmark_dir)
            
    def set_rgb(self, rgb):
        self.curr_rgb = rgb
    
    def set_semantic(self, semantic):
        self.curr_semantic = semantic
        
    def set_depth(self, depth):
        self.curr_depth = depth
    
    def add_detection(self, detection):
        # process detection to desired yolo format with timestamp tag
        detection_dict = {
            "img_id": f"{self.step:04}",
            "detections": [] # list of dict of {id, x_center, y_center, w, h}
        }
        for i in range(len(detection.xyxy)):
            xyxy = detection.xyxy[i]
            class_id = detection.class_id[i]
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            
            # normalize to 0 ~ 1
            x_center /= self.curr_rgb.shape[1]
            y_center /= self.curr_rgb.shape[0]
            w /= self.curr_rgb.shape[1]
            h /= self.curr_rgb.shape[0]
            
            # append to detection dict
            detection_dict["detections"].append({
                "id": int(class_id),
                "x_center": x_center,
                "y_center": y_center,
                "w": w,
                "h": h
            })
        # self.detection_list.append(detection)
        self.curr_detection = detection_dict
    
    def add_pose(self, pose):
        # process pose 
        pose = pose.flatten() # (16,)
        self.pose_list.append(pose)
    
    def save_perception_benchmark(self):
        rgb_dir = os.path.join(self.perception_benchmark_dir, "rgb")
        depth_dir = os.path.join(self.perception_benchmark_dir, "depth")
        detection_dir = os.path.join(self.perception_benchmark_dir, "detection")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(detection_dir, exist_ok=True)
        # save rgb and depth
        cv2.imwrite(os.path.join(rgb_dir, f"{self.step:04}.png"), self.curr_rgb)
        np.save(os.path.join(depth_dir, f"{self.step:04}.npy"), self.curr_depth)
        # save detection
        with open(os.path.join(detection_dir, f"{self.step:04}.json"), 'w') as f:
            json.dump(self.curr_detection, f, indent=4)
        # save pose
        np.save(os.path.join(self.perception_benchmark_dir, "pose.npy"), np.array(self.pose_list))
        
        # save semantic
        if self.curr_semantic is not None:
            semantic_dir = os.path.join(self.perception_benchmark_dir, "semantic")
            os.makedirs(semantic_dir, exist_ok=True)
            np.save(os.path.join(semantic_dir, f"{self.step:04}.npy"), self.curr_semantic)
        
    ### --- Visualization logging --- ###
    def new_image(self, h, w):
        self.visualize_image = np.full((h, w, 3), 255, dtype=np.uint8) 
    
    def add_entropy_img(self, agent_coordinate):
        # occupancy_map = crop_around_point((paper_map_trans.permute(1, 2, 0) * 255).numpy().astype(np.uint8), agent_coordinate, (150, 200))
        entropy_map_positions = {"distribution": (540, 20), "entropy": (540, 240)}
        for title, entropy_map in self.entropy_maps.items():
            position = entropy_map_positions[title]
            add_resized_image(self.visualize_image, entropy_map, (position[0], position[1]), (200, 200))
            add_text(self.visualize_image, title, (position[0]+10, position[1]), font_scale=0.3, thickness=1)
            
    def save_image(self, filename, with_step=False):
        if with_step:
            filename = f"{filename.split('.')[0]}_{self.step}.png"
        visualize_image_bgr = cv2.cvtColor(self.visualize_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, visualize_image_bgr)
        
        # visualize_image_bgr = visualize_image_bgr[:, :, ::-1]
        self.visualize_history.append(visualize_image_bgr)
    
    def view_full_map_raw(self, full_map : np.ndarray):
        full_map = full_map[0,0].cpu().numpy() # 0 ~ 1
        color_map = plt.get_cmap('viridis')
        full_map_rgb = color_map(full_map)[...,:3] * 255
        full_map_rgb = np.flipud(full_map_rgb).astype(np.uint8)
        
        add_text(self.visualize_image, "Full Map", (100, 50))
        add_resized_image(self.visualize_image, full_map_rgb, (100, 100), (800, 800))
    
    def view_fbe_free_map_raw(self, fbe_free_map : np.ndarray):
        fbe_free_map = fbe_free_map[0,0].cpu().numpy() # 0 ~ 1
        color_map = plt.get_cmap('viridis')
        fbe_free_map_rgb = color_map(fbe_free_map)[...,:3] * 255
        fbe_free_map_rgb = np.flipud(fbe_free_map_rgb).astype(np.uint8)
        
        add_text(self.visualize_image, "fbe_free_map", (1000, 50))
        add_resized_image(self.visualize_image, fbe_free_map_rgb, (1000, 100), (800, 800))
    
    def view_frontier_map_raw(self, frontier_map : np.ndarray):
        frontier_map = frontier_map.cpu().numpy() # 0 ~ 1
        color_map = plt.get_cmap('viridis')
        frontier_map_rgb = color_map(frontier_map)[...,:3] * 255
        frontier_map_rgb = np.flipud(frontier_map_rgb).astype(np.uint8)
        
        add_text(self.visualize_image, "frontier_map", (1000, 950))
        add_resized_image(self.visualize_image, frontier_map_rgb, (1000, 1000), (800, 800))
    
    def view_occupancy_map_raw(self, occupancy_map, save = None):
        self.occupancy_rgb = occupancy_map.permute(1,2,0).cpu().numpy()
        self.occupancy_rgb = (self.occupancy_rgb * 255).astype(np.uint8)
        if save is not None:
            img = cv2.cvtColor(self.occupancy_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save, img)
        else:
            add_text(self.visualize_image, "occupancy_map", (100, 950))
            add_resized_image(self.visualize_image, occupancy_map, (100, 1000))
    
    def view_all_maps(self, mapping_results):
        self.view_full_map_raw(mapping_results.full_map)
        self.view_fbe_free_map_raw(mapping_results.fbe_free_map)
        self.view_frontier_map_raw(mapping_results.frontier_map)
        self.view_occupancy_map_raw(mapping_results.occupancy_map)
    
    
    def view_entropy_map(self, entropy_map : np.ndarray, title : str, position : tuple = (0,0), reduce_size = 1, save=None):
        '''
        position: (x, y) in image coordinate
        '''
        color_map = plt.get_cmap('hot')
        entropy_map_rgb = color_map(entropy_map)[...,:3] * 255
        entropy_map_rgb = entropy_map_rgb.astype(np.uint8)
        entropy_map_rgb = np.flipud(entropy_map_rgb)
        # blend in with occupancy map
        # alpha = 0.6
        # beta = 1 - alpha
        # blended_map = cv2.addWeighted(entropy_map_rgb, alpha, self.occupancy_rgb, beta, 0)
        # entropy_map_rgb = entropy_map_rgb.astype(np.uint8)
        
        # add axis on the side
        fig_shape = entropy_map.shape
        fig, ax = plt.subplots(figsize=(fig_shape[0]/100/reduce_size, fig_shape[0]/100/reduce_size))
        # overlay the entropy map on the occupancy map
        # ax.imshow(blended_map, origin='lower')
        
        # entropy with color bar
        im = ax.imshow(np.flipud(entropy_map), cmap='inferno') # this aligns with visuals
        # im = ax.imshow(entropy_map, cmap='inferno', origin='lower')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        # darw to save buffer
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        ncols, nrows = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)[..., :3]
        
        if save is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save, image)
        else:
            # resize = (image.shape[0]//reduce_size, image.shape[1]//reduce_size)
            self.entropy_maps[title] = image
        plt.close(fig)
        
    def view_image_with_bbox(self, rgb, bboxes, captions, confidence_scores, color, save=None):
        img_w_bbox = draw_bbox_with_label(rgb, bboxes, captions, confidence_scores, color)
        # img_w_bbox_bgr = cv2.cvtColor(img_w_bbox, cv2.COLOR_RGB2BGR)
        
        if save is not None:
            cv2.imwrite(save, img_w_bbox)
        else:
            img_w_bbox = cv2.cvtColor(img_w_bbox, cv2.COLOR_BGR2RGB)
            add_resized_image(self.visualize_image, img_w_bbox, (50, 50))
    
    def visualize_scenegraph(self, llm, selected_frontier_id):
        """
        Visualizes the scenegraph on the given image by drawing connections and captions
        for rooms, groups, and objects.

        Parameters:
            image (np.ndarray): The input image on which the scenegraph will be visualized.

        Returns:
            np.ndarray: The image with the visualized scenegraph.
        """
        # image = draw_scenegraph(map, self.memory_module)
        llm_name = '_'.join(llm.split(':'))
        sg_img_save_path = f"dump/scenegraph_{llm_name}/{self.experiment_name}/{self.scene}/{self.episode}/{self.step}_{selected_frontier_id}.png"
        if os.path.exists(sg_img_save_path):
            sg_image = cv2.imread(sg_img_save_path)
            sg_image = cv2.resize(sg_image, (800, 400))
            sg_image = cv2.cvtColor(sg_image, cv2.COLOR_BGR2RGB)
            self.sg_image = sg_image
        add_resized_image(self.visualize_image, self.sg_image, (800, 20), (800, 400))
    
    def update_bev_scenegraph(self, image):
        if image is None:
            self.sg_image = None
        else:   
            self.sg_image = image[:,:,0:3][:,:,::-1] # remove alpha and convert to BGR
    
    def update_pred_bev_scenegraph(self, image):
        if image is None:
            self.sg_pred_image = None
        else:
            self.sg_pred_image = image[:,:,0:3][:,:,::-1] # remove alpha and convert to BGR

    def copy_visualize_image(self, visualize_image, position = (0,0)):
        add_resized_image(self.visualize_image, visualize_image, (position[0], position[1]))

    def increase_function_call_count(self, function_name, args=None):
        """
        Increases the function call count for a given function name and its arguments.
        @param function_name: it doesn't have to be the exact function name, pick a name if you want
        @param args: The arguments you want to log. Format: {'arg1': value1, 'arg2': value2, ...}
        """
        if function_name not in self.function_call_history:
            self.function_call_history[function_name] = {'count':0, 'args':{}}
        if args is not None:
            for key, value in args.items():
                if key not in self.function_call_history[function_name]['args']:
                    self.function_call_history[function_name]['args'][key] = {value: 0}
                if value not in self.function_call_history[function_name]['args'][key]:
                    self.function_call_history[function_name]['args'][key][value] = 0
                self.function_call_history[function_name]['args'][key][value] += 1
        self.function_call_history[function_name]['count'] += 1

    def get_function_call_count(self, function_name, arg_name=None, arg_value=None):
        """
        Returns the function call count for a given function name and its arguments.
        @param arg_name & arg_value: The argument name  and value you want to check.
                                     If either is None, return the total count.
        """
        if function_name not in self.function_call_history:
            return 0
        if arg_name is not None and arg_value is not None:
            if arg_name not in self.function_call_history[function_name]['args']:
                return 0
            if arg_value not in self.function_call_history[function_name]['args'][arg_name]:
                return 0
            return self.function_call_history[function_name]['args'][arg_name][arg_value]
        return self.function_call_history[function_name]['count']
        

vln_logger = VLNLogger(
    name="VLN", level=logging.INFO, format_str="[%(asctime)-15s]:[%(name)s]:[%(module)s] %(message)s"
)