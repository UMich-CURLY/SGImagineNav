# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/dense_slam_gui.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import os
import numpy as np
import threading
import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont



def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable

class ReconstructionWindow:

    def __init__(self, args, font_id, send_queue, receive_queue):
        self.args = args

        self.device = "cuda:{}".format(self.args.gpu_id)

        self.window = gui.Application.instance.create_window(
            'Open3D - Reconstruction', 1280, 800)

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Items in adjustable props
        self.adjustable_prop_grid = gui.VGrid(2, spacing,
                                              gui.Margins(em, 0, em, 0))

        ### Update surface?
        rgb_pc_label = gui.Label('RGB PC?')
        self.rgb_pc_box = gui.Checkbox('')
        self.rgb_pc_box.checked = True
        self.adjustable_prop_grid.add_child(rgb_pc_label)
        self.adjustable_prop_grid.add_child(self.rgb_pc_box)

        ### Show trajectory?
        trajectory_label = gui.Label('Trajectory?')
        self.trajectory_box = gui.Checkbox('')
        self.trajectory_box.checked = True
        self.adjustable_prop_grid.add_child(trajectory_label)
        self.adjustable_prop_grid.add_child(self.trajectory_box)

        ## Application control
        b = gui.ToggleSwitch('Resume/Pause')
        b.set_on_clicked(self._on_switch)

        self.text_edit = gui.TextEdit()
        self.submit_button = gui.Button("Submit")
        self.submit_button.set_on_clicked(self._on_submit)
        self.input_text = None

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()


        ### Rendered image tab
        tab2 = gui.Vert(0, tab_margins)
        self.semantic_image = gui.ImageWidget()
        self.map_image = gui.ImageWidget()
        tab2.add_child(self.semantic_image)
        tab2.add_fixed(vspacing)
        tab2.add_child(self.map_image)
        tabs.add_tab('Semantic images', tab2)

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_color_image = gui.ImageWidget()
        self.input_depth_image = gui.ImageWidget()
        tab1.add_child(self.input_color_image)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.input_depth_image)
        tabs.add_tab('Input images', tab1)

        ### Info tab
        tab3 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id
        tab3.add_child(self.output_info)
        tabs.add_tab('Info', tab3)

        self.panel.add_child(gui.Label('Reconstruction settings'))
        self.panel.add_child(self.adjustable_prop_grid)
        self.panel.add_child(self.text_edit)
        self.panel.add_child(self.submit_button)
        self.panel.add_child(b)
        self.panel.add_stretch()
        self.panel.add_child(tabs)


        # Scene widget
        self.widget3d = gui.SceneWidget()

        # FPS panel
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.fps_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.saved_objects = None
        self.saved_full_points = None
        self.saved_full_colors = None

        self.is_done = False

        self.is_started = False
        self.is_running = True
        self.episode_n = 0
        self.candidate_id = 0

        self.idx = 0
        self.traj = []
        self.full_pcd_poinst = []
        self.full_pcd_colors = []

        # Start running
        self.send_queue = send_queue
        self.receive_queue = receive_queue
        threading.Thread(name='UpdateMain', target=self.update_main).start()

    def _on_submit(self):
        input_text = self.text_edit.text_value
        print("Input text:", input_text)
        self.send_queue.put([input_text, self.is_running])

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
                                        rect.y, fps_panel_width,
                                        fps_panel_height)

    # Toggle callback: application's main controller
    def _on_switch(self, is_on):
        
        self.is_running = not self.is_running
        self.send_queue.put([None, self.is_running])

    # On start: point cloud buffer and model initialization.
    def _on_start(self):
        max_points = self.est_point_count_slider.int_value

        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32)))
        pcd_placeholder.point.colors = o3c.Tensor(
            np.zeros((max_points, 3), dtype=np.float32))
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)

        # self.model = o3d.t.pipelines.slam.Model(
        #     self.voxel_size_slider.double_value, 16,
        #     self.est_block_count_slider.int_value, o3c.Tensor(np.eye(4)),
        #     o3c.Device(self.device))
        self.is_started = True

        set_enabled(self.fixed_prop_grid, False)
        set_enabled(self.adjustable_prop_grid, True)

    def _on_close(self):
        self.is_done = True

        # if self.is_started:
        print('Saving model to {}...'.format(self.args.path_npz))
        if not os.path.exists(self.args.path_npz):
            os.makedirs(self.args.path_npz)
        np.savez(self.args.path_npz + 'arrays.npz', objects=self.saved_objects, full_points=self.saved_full_points, full_colors=self.saved_full_colors)
        print('Finished.')

            # mesh_fname = '.'.join(self.args.path_npz.split('.')[:-1]) + '.ply'
            # print('Extracting and saving mesh to {}...'.format(mesh_fname))
            # mesh = extract_trianglemesh(self.model.voxel_grid, config,
            #                             mesh_fname)
            # print('Finished.')

            # log_fname = '.'.join(self.args.path_npz.split('.')[:-1]) + '.log'
            # print('Saving trajectory to {}...'.format(log_fname))
            # save_poses(log_fname, self.poses)
            # print('Finished.')

        return True
    
    def matrix2lineset(self, poses, color=[0, 0, 1]):
        '''
        Create a open3d line set from a batch of poses

        poses: (N, 4, 4)
        color: (3,)
        '''
        N = poses.shape[0]
        lineset = o3d.geometry.LineSet()
        if np.all(np.ptp(poses[:, :3, 3], axis=0) == 0):
            return lineset
        
        lineset.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
        lineset.lines = o3d.utility.Vector2iVector(
            np.array([[i, i + 1] for i in range(N - 1)])
        )
        lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
        return lineset
    
    def poses2lineset(self, poses, color=[0, 0, 1]):
        '''
        Create a open3d line set from a batch of poses

        poses: (N, 4, 4)
        color: (3,)
        '''
        N = poses.shape[0]
        lineset = o3d.geometry.LineSet()
        if np.all(np.ptp(poses, axis=0) == 0):
            return lineset
        
        lineset.points = o3d.utility.Vector3dVector(poses)
        lineset.lines = o3d.utility.Vector2iVector(
            np.array([[i, i + 1] for i in range(N - 1)])
        )
        lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
        return lineset


    def init_render(self):
       
        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(90, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [-3, 4, 0], [0, 1, 0])

        # self.widget3d.setup_camera(90, bbox, [camera_matrix[3,0], camera_matrix[3,1], camera_matrix[3,2]])[0, 0, 0]
        # self.widget3d.look_at(camera_matrix[:3,0], camera_matrix[:3,1], camera_matrix[:3,2])
        points = np.random.rand(100, 3)
        colors = np.zeros((100, 3))
        colors[:, 0] = 1  # 红色
        pcd_t = o3d.t.geometry.PointCloud(
                    o3c.Tensor(points.astype(np.float32)))
        pcd_t.point.colors = o3c.Tensor(colors)
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        self.widget3d.scene.add_geometry('points', pcd_t, material)  # Add material argument

        # Add a coordinate frame
        self.widget3d.scene.show_axes(True)

    def render_top_view_map(self, 
                      objects, 
                      point_sum_points,
                      point_sum_colors,
                      candidate_id):
        
        self.widget3d.scene.clear_geometry()
        if self.rgb_pc_box.checked:
            # self.full_pcd_poinst.append(point_sum_points)
            # self.full_pcd_colors.append(point_sum_colors)
            
            # point_sum_points = np.vstack(self.full_pcd_poinst)
            # point_sum_colors = np.vstack(self.full_pcd_colors)
            
            full_pcd = o3d.t.geometry.PointCloud(
            o3c.Tensor(point_sum_points.astype(np.float32)))
            full_pcd.point.colors = o3c.Tensor(point_sum_colors.astype(np.float32))
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.widget3d.scene.add_geometry('full_pcd', full_pcd, material)  # Add material argument


        if len(objects) > 0:
   
            bbox = o3d.geometry.AxisAlignedBoundingBox(objects[candidate_id]['bbox_np'][0].astype(np.float32), objects[candidate_id]['bbox_np'][1].astype(np.float32)) 
            mat = rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = 5.0

            bbox.color = [1, 0, 0]
            self.widget3d.scene.add_geometry(f"bbox_{candidate_id}", bbox, mat)
            # 保存图片
            target_position = bbox.get_center()
            self.widget3d.look_at(target_position+[0, 0, 0], target_position+[-0.5, 4, 0], [0, 1, 0])
            self.candidate_id = candidate_id
            # print('send_image')
            while True:
                try:
                    self.widget3d.scene.scene.render_to_image(self.on_image_rendered)
                    break  # 成功后跳出循环
                except Exception as e:
                    print(f"Error rendering image, retrying: {e}")
            
        
    def update_render(self, 
                      input_depth, 
                      input_color, 
                      semantic_image, 
                      objects, 
                      frustum,
                      point_sum_points,
                      point_sum_colors,
                      traj,
                      plan_path,
                      vis_image,
                      Open3d_goal_pose,
                      candidate_id):
        
        # update view
        # self.widget3d.look_at(camera_position+[0, 0, 0], camera_position+[-0.5, 6, 0], [0, 1, 0])
        
        self.input_depth_image.update_image(
            input_depth.colorize_depth(
                1000.0, self.args.min_depth,
                self.args.max_depth).to_legacy())
        self.input_color_image.update_image(input_color.to_legacy())
        self.semantic_image.update_image(semantic_image.to_legacy())
        # check if type is numpy.ndarray
        if isinstance(vis_image, np.ndarray):
            self.map_image.update_image(vis_image)
        else:
            self.map_image.update_image(vis_image.to_legacy())

        # add the camera
        self.widget3d.scene.remove_geometry("frustum")
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5.0
        self.widget3d.scene.add_geometry("frustum", frustum, mat)

        # add the trajectory
        self.widget3d.scene.remove_geometry("trajectory")
        if self.trajectory_box.checked:
            traj_lineset = self.matrix2lineset(np.stack(traj), color=[0, 1., 0])
            if traj_lineset.has_lines() and traj_lineset.has_points(): 
                mat = rendering.MaterialRecord()
                mat.shader = "unlitLine"
                mat.line_width = 10.0
                self.widget3d.scene.add_geometry("trajectory", traj_lineset, mat)

        self.widget3d.scene.remove_geometry("path_points")
        if len(plan_path) > 0:
            path_lineset = self.poses2lineset(np.stack(plan_path), color=[1., 0, 0])
            if path_lineset.has_lines() and path_lineset.has_points():
                material = rendering.MaterialRecord()
                material.shader = "unlitLine"
                material.line_width = 5.0
                self.widget3d.scene.add_geometry('path_points', path_lineset, material)

        self.widget3d.scene.remove_geometry("full_pcd")
        if self.rgb_pc_box.checked:
            # self.full_pcd_poinst.append(point_sum_points)
            # self.full_pcd_colors.append(point_sum_colors)
            
            # point_sum_points = np.vstack(self.full_pcd_poinst)
            # point_sum_colors = np.vstack(self.full_pcd_colors)
            
            full_pcd = o3d.t.geometry.PointCloud(
            o3c.Tensor(point_sum_points.astype(np.float32)))
            full_pcd.point.colors = o3c.Tensor(point_sum_colors.astype(np.float32))
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.widget3d.scene.add_geometry('full_pcd', full_pcd, material)  # Add material argument

        if len(objects) > 0:
            pcd_sum = []
            pcd_color_sum = []
            bbox_sum = []
            for obj in objects:
                pcd_sum.append(obj['pcd_np'].astype(np.float32))
                pcd_color_sum.append(obj['pcd_color_np'].astype(np.float32))
                bbox_sum.append((obj['bbox_np'][0].astype(np.float32), obj['bbox_np'][1].astype(np.float32)))
            # print("bbox_sum: ", bbox_sum)
            pcd = o3d.t.geometry.PointCloud(
                o3c.Tensor(np.vstack(pcd_sum)))
            pcd.point.colors = o3c.Tensor(np.vstack(pcd_color_sum))
            

            self.widget3d.scene.remove_geometry("points")
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.widget3d.scene.add_geometry('points', pcd, material)  # Add material argument
            
            while True:
                try:
                    self.widget3d.scene.scene.render_to_image(self.on_top_rendered)
                    break  # 成功后跳出循环
                except Exception as e:
                    print(f"Error rendering image, retrying: {e}")


            # bboxs = [o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound) for min_bound, max_bound in bbox_sum]
            # mat = rendering.MaterialRecord()
            # mat.shader = "unlitLine"
            # mat.line_width = 5.0

            
            # for i, bbox in enumerate(bboxs):
            #     # self.widget3d.scene.remove_geometry(f"bbox_{i}")
            #     if i in candidate_id:
                        
            #         bbox.color = [1, 0, 0]
            #         self.widget3d.scene.add_geometry(f"bbox_{i}", bbox, mat)
            #         # 保存图片
            #         target_position = bbox.get_center()
            #         self.widget3d.look_at(target_position+[0, 0, 0], target_position+[-0.5, 4, 0], [0, 1, 0])
            #         self.candidate_id = i
            #         print(self.candidate_id)
            #         self.widget3d.scene.scene.render_to_image(self.on_image_rendered)
            #         time.sleep(0.1)
            #         self.widget3d.scene.remove_geometry(f"bbox_{i}")





        # self.widget3d.scene.remove_geometry("goal_pose")
        # goal_pcd = o3d.geometry.PointCloud()
        # goal_pcd.points = o3d.utility.Vector3dVector(np.array([Open3d_goal_pose]))
        # # Set a large size for the point
        # goal_material = rendering.MaterialRecord()
        # goal_material.shader = "defaultUnlit"
        # goal_material.point_size = 20.0  # Adjust this value to make the point larger
        # # Set the color of the point to distinguish it (e.g., red)
        # goal_colors = np.array([[1.0, 0.0, 0.0]])  # Red color
        # goal_pcd.colors = o3d.utility.Vector3dVector(goal_colors)
        # # Add the goal pose point cloud to the scene
        # self.widget3d.scene.add_geometry("goal_pose", goal_pcd, goal_material)


    def on_top_rendered(self, image):
        
        dump_dir = "{}/{}/episodes/{}/top_render".format(self.args.dump_location,
                                    self.args.exp_name, self.episode_label)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        

        # # Convert the open3d.cuda.pybind.geometry.Image object to a numpy array
        np_image = np.asarray(image)
        # cv2.imwrite(image_filename, np_image)

        # Convert the numpy array to a PIL Image object
        Image.fromarray(np_image)
        
        # # add the number on the image
        # # Initialize drawing context
        # draw = ImageDraw.Draw(pil_image)
        # # Define the text to be added and its position
        # font_size = 40

        # try:
        #     # Attempt to use a truetype font if available
        #     font = ImageFont.truetype("arial.ttf", font_size)
        # except IOError:
        #     # If the truetype font is not available, use the default PIL font
        #     font = ImageFont.load_default(font_size)

        # # Calculate text size to center it
        # # bbox = draw.textbbox((0, 0), text, font=font)
        # text_width = 45
        # text_height = 45
        # padding = 3
        # position = (3, 3)  # Adjust position as needed

        # # Define the rectangle coordinates
        # rect_x0 = position[0] - padding
        # rect_y0 = position[1] - padding
        # rect_x1 = position[0] + text_width + padding
        # rect_y1 = position[1] + text_height + padding

        # # Draw the white rectangle
        # draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="white")

        # # Add text to image
        # draw.text(position, str(self.idx), fill="red", font=font)

        # Save the image using PIL
        # pil_image.save(image_filename)
        # print(f"Saved screenshot to {image_filename}")





    def on_image_rendered(self, image):
        
        dump_dir = "{}/{}/episodes/{}/image_render".format(self.args.dump_location,
                                    self.args.exp_name, self.episode_label)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        # image_filename = '{}/Vis-{}-{}.png'.format(dump_dir, self.idx, self.candidate_id)

        # Convert the open3d.cuda.pybind.geometry.Image object to a numpy array
        np_image = np.asarray(image)
        resized_image1 = cv2.resize(np_image, (480, 480))
        
        candidate_image = np.asarray(self.candidate_image)
        resized_image2 = cv2.resize(candidate_image, (480, 480))
        
        # Combine images horizontally
        combined_image = np.hstack((resized_image1, resized_image2))
        
        # Convert the numpy array to a PIL Image object
        pil_image = Image.fromarray(combined_image)
        
        # add the number on the image
        # Initialize drawing context
        draw = ImageDraw.Draw(pil_image)
        # Define the text to be added and its position
        font_size = 40

        try:
            # Attempt to use a truetype font if available
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            # If the truetype font is not available, use the default PIL font
            font = ImageFont.load_default(font_size)

        # Calculate text size to center it
        # bbox = draw.textbbox((0, 0), text, font=font)
        text_width = 45
        text_height = 45
        padding = 3
        position = (3, 3)  # Adjust position as needed

        # Define the rectangle coordinates
        rect_x0 = position[0] - padding
        rect_y0 = position[1] - padding
        rect_x1 = position[0] + text_width + padding
        rect_y1 = position[1] + text_height + padding

        # Draw the white rectangle
        draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="white")

        # Add text to image
        draw.text(position, str(self.candidate_id), fill="red", font=font)

        # Save the image using PIL
        # pil_image.save(image_filename)
        # print(f"Saved screenshot to {image_filename}")

        # print(self.candidate_id)
        # print('render: %.3f秒'%ss_time) 
        self.candidate_id = -1

    # Major loop
    def update_main(self):
        
        height = self.args.frame_height
        width = self.args.frame_width
        
        cx = (width - 1.) / 2.
        cy = (height - 1.) / 2.
        fx = (width / 2.) / np.tan(np.deg2rad(self.args.hfov / 2.))
        fy = (height / 2.) / np.tan(np.deg2rad(self.args.hfov / 2.))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix,
                               o3d.core.Dtype.Float64)

        o3d.core.Device(self.device)

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render())

        fps_interval_len = 1
        self.idx = 0
        # pcd = None

        start = time.time()
        while not self.is_done:
            if not self.receive_queue.empty():
                image_rgb, image_depth, annotated_image, objects, point_sum_points, point_sum_colors, traj, episode_n, episode_label, plan_path, vis_image, Open3d_goal_pose, time_step_info, candidate_id = self.receive_queue.get()
                
                if episode_n > self.episode_n:
                    self.idx = 0
                    self.widget3d.scene.clear_geometry()
                    self.episode_n = episode_n
                    self.episode_label = episode_label
                    self.full_pcd_poinst = []
                    self.full_pcd_colors = []
                           
                if len(candidate_id) == 0:
                    
                    self.saved_objects = objects
                    # self.saved_full_points = point_sum_points
                    # self.saved_full_colors = point_sum_colors

                    image_depth = (image_depth * 1000).astype(np.uint16)
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) 

                    depth = o3d.t.geometry.Image(image_depth)
                    color = o3d.t.geometry.Image(image_rgb)
                    semantic = o3d.t.geometry.Image(annotated_image)
                    vis_image = o3d.t.geometry.Image(vis_image)

                    T_frame_to_model = o3c.Tensor(traj[-1])
                    frustum = o3d.geometry.LineSet.create_camera_visualization(
                        color.columns, color.rows, intrinsic.numpy(),
                        np.linalg.inv(T_frame_to_model.cpu().numpy()), 0.2)
                    frustum.paint_uniform_color([0.961, 0.475, 0.000])

                    torch.cuda.synchronize()


                    # self.traj.append(camera_matrix)

        
                    # Output FPS
                    if (self.idx % fps_interval_len == 0):
                        end = time.time()
                        elapsed = end - start
                        start = time.time()
                        self.output_fps.text = 'FPS: {:.3f}'.format(fps_interval_len /
                                                                    elapsed)

                    # Output info
                    info = 'Frame {}/{}\n\n'.format(self.idx, 500)
                    info += 'Transformation:\n{}\n\n'.format(
                        np.array2string(T_frame_to_model.numpy(),
                                        precision=3,
                                        max_line_width=40,
                                        suppress_small=True,
                                        formatter={'float_kind': lambda x: f"{x:.2f}"}))
                    info += time_step_info

                    self.output_info.text = info

                    gui.Application.instance.post_to_main_thread(
                        self.window, lambda: self.update_render(
                            depth,
                            color,
                            semantic, 
                            objects, 
                            frustum,
                            point_sum_points,
                            point_sum_colors,
                            traj,
                            plan_path,
                            vis_image,
                            Open3d_goal_pose,
                            candidate_id)
                            )
                    self.idx += 1
                    
                else:
                    
                    for index, i in enumerate(candidate_id):
                        self.candidate_id = 0
                        self.candidate_image = image_rgb[index]
                        gui.Application.instance.post_to_main_thread(
                            self.window, lambda i = i: self.render_top_view_map(
                                objects, 
                                point_sum_points,
                                point_sum_colors,
                                i)
                            )
                        while (self.candidate_id != -1):
                            time.sleep(0.01)
            # else:
            #     gui.Application.instance.post_to_main_thread(
            #         self.window, lambda: self.update_render(
            #             depth,
            #             color,
            #             semantic, 
            #             objects, 
            #             frustum,
            #             point_sum_points,
            #             point_sum_colors,
            #             traj,
            #             plan_path,
            #             vis_image,
            #             Open3d_goal_pose,
            #             candidate_id)
            #             )
                        


            time.sleep(0.1)

