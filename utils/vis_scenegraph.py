import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import gc
import sys
import cv2
import shutil
from matplotlib.path import Path
from mpl_toolkits.mplot3d import proj3d
import matplotlib.transforms as mtransforms
from skimage.transform import AffineTransform
from utils.image_process import add_resized_image
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from utils.scene_graph_utils import get_sorted_key_value_lists


def crop_around_point(point, image=(800, 800), size=(300, 300)):
    img_height, img_width = image
    crop_width, crop_height = size
    point = [int(x) for x in point]
    
    left = max(point[0] - crop_width // 2, 0)
    top = max(point[1] - crop_height // 2, 0)
    right = min(point[0] + (crop_width - crop_width // 2), img_width)
    bottom = min(point[1] + (crop_height - crop_height // 2), img_height)
    
    if right - left < crop_width:
        if left == 0:
            right = left + crop_width
        else:
            left = right - crop_width
    if bottom - top < crop_height:
        if top == 0:
            bottom = top + crop_height
        else:
            top = bottom - crop_height
    
    return top, bottom, left, right

def load_json(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
    return data

def get_z_for_node(node):
    """
    Assign a z-coordinate based on the node's hierarchical type.
    Higher levels get higher z-values.
    """
    t = node.get("type", "object")
    if t == "room":
        return 20
    elif t == "region":
        return 15
    else:
        return 0

def extract_nodes(scene_graph):
    """
    Extracts nodes from a scene graph into a flat dictionary.
    Each node is keyed by its unique id (as a string) and includes:
      - caption, center (x, y), type, and a z-level determined by type.
    """
    nodes = {}
    for room_id, room in enumerate(scene_graph.get("rooms", [])):
        if len(room["regions"]) == 0:
            continue
        nodes[f'r{room_id}'] = {
            "id": room_id,
            "caption": room.get("caption", f"Room {room_id}"),
            "center": room.get("center", [0, 0]),
            "type": "room",
            "z": 20
        }
        # Room-level objects.
        for obj_id, obj in enumerate(room.get("objects", [])):
            obj_id = str(obj_id)
            nodes[f'r{room_id}_o{obj_id}'] = {
                "id": obj_id,
                "caption": obj.get("caption", f"Object {obj_id}"),
                "center": obj.get("center", room.get("center", [0, 0])),
                "type": "object",
                "z": 0
            }
        # Regions and their objects.
        for region_id, region in enumerate(room.get("regions", [])):
            nodes[f'r{room_id}_r{region_id}'] = {
                "id": region_id,
                "caption": region.get("caption", f"Region {region_id}")[:50],
                "center": region.get("center", room.get("center", [0, 0])),
                "type": "region",
                "z": 15
            }
            for obj_id, obj in enumerate(region.get("objects", [])):
                nodes[f'r{room_id}_r{region_id}_o{obj_id}'] = {
                    "id": obj_id,
                    "caption": obj.get("caption", f"Object {obj_id}"),
                    "center": obj.get("center", region.get("center", room.get("center", [0, 0]))),
                    "type": "object",
                    "z": 0
                }
    return nodes

def extract_edges_3d(scene_graph, edge_distance_threshold=50):
    """
    Extracts hierarchical edges from a scene graph and returns them as a list of tuples:
      (start_point, end_point, edge_type)
    Each start_point and end_point is a 3D coordinate [x, y, z]. The edge is only included if the
    horizontal (x, y) distance between parent and child is less than the threshold.
    """
    edges = []
    for room in scene_graph.get("rooms", []):
        if "center" in room and room["center"] == '':
            continue
        room_center = room.get("center", [0, 0])
        room_z = 20
        [room_center[0], room_center[1], room_z]
        # Room → regions.
        for region in room.get("regions", []):
            region_center = region.get("center", room_center)
            region_z = 15
            region_center_3d = [region_center[0], region_center[1], region_z]
            # edges.append((room_center_3d, region_center_3d, "room_region"))
            # Region → region objects.
            for obj in region.get("objects", []):
                obj_center = obj.get("center", region_center)
                obj_z = 0
                obj_center_3d = [obj_center[0], obj_center[1], obj_z]
                edges.append((region_center_3d, obj_center_3d, "region_object"))
    return edges


def warp(T1, T2):
    """
    Return an affine transform that warp triangle T1 into triangle T2.
    Raises
    ------
    `LinAlgError` if T1 or T2 are degenerated triangles
    """
    transform = AffineTransform()
    transform.estimate(T1,T2)
    return mtransforms.Affine2D(np.array(transform))

def imshow3d(ax,img,XYZ=None,x = 0,y=0):
    h,w = img.shape[:2]
    aspect = (w,h,(w+h)/3)
    ax.set_box_aspect(aspect)
    ax.set_proj_type('ortho')
    UV = np.array([[0,0],
                   [w,0],
                   [w,h],
                   [0,h]])
    if(XYZ==None):
        XYZ = [(x,y,0),(x+w,y,0),(x+w,y+h,0),(x,y+h,0)]
    xy = []
    for xyz in XYZ:
        x,y,z = np.array(proj3d.proj_transform(*xyz, ax.get_proj()))
        # print(ax.get_proj())
        # print(f'{x} {y} {z}')
        xy.append((x,y))
    xy = np.array(xy)
    transform = warp(UV,xy) + ax.transData
    path =  Path([UV[0], UV[1], UV[2], UV[3],UV[0]], closed=True)
    ax.imshow(img, interpolation=None, origin='lower',
                           transform=transform, clip_path=(path,transform),aspect=None)
    ax.set_box_aspect(aspect)

def imshow3d_fast(ax, array, value_direction='z', pos=0, norm=None, cmap=None, stride=2, alpha = 0.1):
    """
    Faster version of imshow3d using strides to reduce mesh resolution.
    """
    if norm is None:
        norm = Normalize()
    # colors = plt.get_cmap(cmap)(norm(array))
    colors = array

    if value_direction == 'x':
        nz, ny, _ = array.shape
        zi, yi = np.mgrid[0:nz + 1:stride, 0:ny + 1:stride]
        xi = np.full_like(yi, pos)
    elif value_direction == 'y':
        nx, nz, _ = array.shape
        xi, zi = np.mgrid[0:nx + 1:stride, 0:nz + 1:stride]
        yi = np.full_like(zi, pos)
    elif value_direction == 'z':
        ny, nx, _ = array.shape
        yi, xi = np.mgrid[0:ny + 1:stride, 0:nx + 1:stride]
        zi = np.full_like(xi, pos)
    else:
        raise ValueError(f"Invalid value_direction: {value_direction!r}")

    transparent = np.zeros([colors.shape[0], colors.shape[1], 1]) + alpha
    colors = np.concatenate([colors, transparent], axis=-1)
    ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, facecolors=colors[::stride, ::stride])


def visualize_scenegraph_diff_with_hierarchy_3d(original, predicted, target_location=None, map=None, agent_location=None, edge_threshold=50, save_path=None):
    """
    3D visualization of scene graph differences with hierarchical connections.
    
    - Nodes are plotted at their spatial (x, y) locations, with a z-level representing hierarchy:
         room: z=20, region: z=10, object: z=0.
    - Marker shapes represent node types:
         room: square ('s'), region: triangle ('^'), object: circle ('o').
    - Common nodes (present in both original and predicted) are grey.
    - New nodes (only in predicted) are red.
    - Removed nodes (only in original) are blue.
    - Only nodes whose horizontal distance (x, y) is within edge_threshold are connected.
      Edge styles vary by relationship:
         Room→Region: dashed blue  
         Room→Object: solid purple  
         Region→Object: dash-dot orange
    - The target location is marked with a black star (at z=0).
    - A custom legend is added that maps node type markers and node colors.
    """
    # Extract nodes and edges.
    def _vis_pair(original, predicted, ax):
        orig_nodes = extract_nodes(original)
        pred_nodes = extract_nodes(predicted)
        edges = extract_edges_3d(predicted, edge_distance_threshold=edge_threshold)
        
        # Determine node sets.
        common_ids = set(orig_nodes.keys()) & set(pred_nodes.keys())
        removed_ids = set(orig_nodes.keys()) - set(pred_nodes.keys())

        # Draw hierarchical edges.
        for edge in edges:
            start, end, etype = edge
            if etype == "room_region":
                linestyle, color = "--", "blue"
            elif etype == "room_object":
                linestyle, color = "-", "purple"
            elif etype == "region_object":
                linestyle, color = "-.", "orange"
            else:
                linestyle, color = "-", "black"
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            ax.plot(xs, ys, zs, color=color, linestyle=linestyle, linewidth=1)
        
        fontsize = 3
        # Plot predicted nodes.
        for node_id, node in pred_nodes.items():
            center = node.get("center", [0, 0])
            z = get_z_for_node(node)
            x, y = center[0], center[1]
            node_type = node.get("type", "object")
            marker = shape_map.get(node_type, "o")
            color = "grey" if node_id in common_ids else "red"
            ax.scatter(x, y, z, color=color, marker=marker, s=2)
            ax.text(x, y, z, f" {node.get('caption','')}", fontsize=fontsize)
        
        # Plot removed nodes.
        for node_id in removed_ids:
            node = orig_nodes[node_id]
            center = node.get("center", [0, 0])
            z = get_z_for_node(node)
            x, y = center[0], center[1]
            marker = shape_map.get(node.get("type", "object"), "o")
            ax.scatter(x, y, z, color="blue", marker=marker, s=2)
            ax.text(x, y, z, f" {node.get('caption','')}", fontsize=fontsize)

    # Define marker shapes by node type.
    shape_map = {
        "room": "s",      # square
        "region": "^",    # triangle
        "object": "o"     # circle
    }
    # Create 3D figure.
    fig = plt.figure(figsize=(6, 5.5))
    if isinstance(original, list) and isinstance(predicted, list) and len(original) == 2:
        axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
    else:
        axes = [fig.add_subplot(111, projection='3d')]

    for i, (ori, pred) in enumerate(zip(original, predicted)):
        ax = axes[i]
        ax.set_proj_type('ortho')
        ax.cla()
        top, bottom, left, right = crop_around_point(agent_location)
        ax.set_xlim(left, right)
        ax.set_ylim(top, bottom)
        # ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_zlim(0, 20)
        if map is not None:
            # imshow3d_fast(ax, map[::-1])
            # imshow3d_fast(ax, map)
            imshow3d(ax, map[top:bottom,left:right], x=left, y=top)
        _vis_pair(ori, pred, ax)
        # Plot target location (assumed at ground level, z=0).
        # frontier: [y, x] (grid coordinate, x: left to right, y: up to down)
        # agent coordinate: [x,y]
        ax.scatter(target_location[1], target_location[0], 0, color="black", marker="*", s=10, label="Target Location")
        ax.scatter(agent_location[0], agent_location[1], 0, color="red", marker="*", s=10, label="Agent Location")
        
        # Build custom legend.
        # Create legend handles for node types (marker shapes).
        markersize = 2
        legend_size = 2
        axis_size = 3
        label_size = 3
        # ax.set_xlabel("X Coordinate", fontsize=axis_size)
        # ax.set_ylabel("Y Coordinate", fontsize=axis_size)
        # ax.set_zlabel("Hierarchical Level (z)", fontsize=axis_size)
        if i == 0:
            ax.set_title("Sub Scene Graph", fontsize=axis_size)
        else:
            ax.set_title("Complete Scene Graph", fontsize=axis_size)
        if i == 0:
            type_handles = [
                Line2D([], [], marker='s', color='w', label='Room', markerfacecolor='grey', markersize=markersize),
                Line2D([], [], marker='^', color='w', label='Region', markerfacecolor='grey', markersize=markersize),
                Line2D([], [], marker='o', color='w', label='Object', markerfacecolor='grey', markersize=markersize),
            ]
            # Create legend handles for node statuses (colors).
            status_handles = [
                Line2D([], [], marker='o', color='w', label='Common', markerfacecolor='grey', markersize=markersize),
                Line2D([], [], marker='o', color='w', label='New', markerfacecolor='red', markersize=markersize),
                Line2D([], [], marker='o', color='w', label='Removed', markerfacecolor='blue', markersize=markersize),
            ]
            # Create handle for target location.
            target_handle = Line2D([], [], marker='*', color='w', label='Frontier Location', markerfacecolor='black', markersize=markersize)
            target_handle = Line2D([], [], marker='*', color='w', label='Agent Location', markerfacecolor='red', markersize=markersize)
            
            # Combine all legend handles.
            legend_handles = type_handles + status_handles + [target_handle]
            ax.legend(handles=legend_handles, loc='upper left', fontsize=legend_size)
        
        # ax.grid(axis="z", linestyle="")
        ax.grid(False)
        ax.tick_params(axis="x", labelsize=label_size, pad=0.1)
        ax.tick_params(axis="y", labelsize=label_size, pad=0.1)
        ax.tick_params(axis="z", labelsize=label_size, pad=0.1)
        ax.set_zticks([])
        ax.set_zticklabels([])
        
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        # print(f"Figure saved to {save_path}")
    plt.close()
    del fig  # Delete figure reference
    gc.collect()
    return

# Below is the BEV map with differnt level with different semantic size
def extract_nodes_bev(scene_graph):
    """
    list of node dicts: {type, caption, center}.
    """
    nodes = []
    for room in scene_graph.get("rooms", []):
        # ROOM node
        room_center = room.get("center", [0,0])
        # nodes.append({
        #     "type": "room",
        #     "caption": room.get("caption", "room"),
        #     "center": room_center
        # })
        # REGIONS
        for region in room.get("regions", []):
            reg_center = region.get("center", room_center)
            if isinstance(region.get("caption", "region"), dict):
                captions, confidences = get_sorted_key_value_lists(region['caption'])
                caption = f'{captions[0]}: {confidences[0]:.2f}'
            else:
                caption = region.get("caption", "region")
            if not region.get("predicted", False):
                nodes.append({
                    "type": "region",
                    "caption": caption,
                    "center": reg_center,
                })
            for obj in region.get("objects", []):
                obj_center = obj.get("center", reg_center)
                nodes.append({
                    "type": "object",
                    "caption": '{}{:.2f}'.format(obj.get("caption", "object"), obj.get("confidence", 0)),
                    "center": obj_center
                    })
    return nodes

def visualize_score(save_path, targets):
    """
    Load the saved BEV image and overlay score text using matplotlib.
    """
    # Load image
    img = mpimg.imread(save_path)

    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis('off')

    # Draw each target's info in top-left corner
    for i, target in enumerate(targets):
        x, y = target['center']
        score = target['score']
        ax.text(x+2, y, str(score))

    # Save or display
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    print(f"Annotated image saved with scores: {save_path}")
    return 

def visualize_BEV(occ_map, scene_graph, save_path="floorplan_style.png", crop_size=(800, 800), target_locations=None, agent_location=None):
    """
    Visualize BEV: crop to center of the map and overlay scene graph.
    """
    if scene_graph is None:
        return None
    fontsize_dict = {
        "room": 25,
        "region": 10,
        "region_predicted": 10,
        "object": 7
    }
    color_dict = {
        "room": 'black',
        "region": 'darkblue',
        "region_predicted": 'darkred',
        "object": 'darkgreen'
    }

    H, W = occ_map.shape[:2]
    # crop_h, crop_w = crop_size

    # # Calculate crop bounds (center crop)
    # start_x = W // 2 - crop_w // 2
    # start_y = H // 2 - crop_h // 2
    # end_x = start_x + crop_w
    # end_y = start_y + crop_h

    # Crop the occupancy map
    # cropped_occ_map = occ_map[start_y:end_y, start_x:end_x]

    # Load scene graph & extract nodes
    nodes = extract_nodes_bev(scene_graph)
    start_x = 0
    start_y = 0
    end_x = W
    end_y = H
    # Plot
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ax.imshow(cropped_occ_map)
    ax.imshow(occ_map[::-1])
    ax.set_aspect('equal')
    ax.axis('off')

    x, y = agent_location
    new_x = H - (x - start_x)
    new_y = y - start_y
    ax.scatter(new_y, new_x, s=50, color="red", marker="*", label="Agent")

    for node in nodes[::-1]:
        center = node.get("center", [])
        if not isinstance(center, (list, tuple, np.ndarray)) or len(center) != 2:
            continue

        x, y = center

        # Skip if outside crop
        if not (start_x <= x <= end_x and start_y <= y <= end_y):
            continue

        # Translate coordinates to cropped image
        new_x = H - (x - start_x)
        new_y = y - start_y

        ax.text(
            new_y, new_x, node["caption"],
            fontsize=fontsize_dict[node["type"]],
            color=color_dict[node["type"]],
            ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=0.7)
        )

    if isinstance(target_locations[0], list) or isinstance(target_locations, np.ndarray) and len(np.shape(target_locations)) == 2:
        for target_i, target_loc in enumerate(target_locations):
            x, y = target_loc
            new_x = H - (x - start_x)
            new_y = y - start_y
            # ax.scatter(new_y, new_x, s=50, color='white', marker="*", label="Target")
            # ax.text(new_y+text_offset[0], new_x+text_offset[1], str(target_i), fontsize=3)
            circle = Circle((new_y, new_x), radius=8, edgecolor='white', facecolor='black', linewidth=1)
            ax.add_patch(circle)
            ax.text(new_y, new_x, str(target_i), color='white', fontsize=12, ha='center', va='center')
    elif isinstance(target_locations[0], dict):
        for target_i, target_loc in enumerate(target_locations):
            x, y = target_loc['center']
            new_x = H - (x - start_x)
            new_y = y - start_y
            # ax.scatter(new_y, new_x, s=50,color='white', marker="*", label="Target")
            # ax.text(new_y+text_offset[0], new_x+text_offset[1], f'{target_i}:{s}', fontsize=3)
            circle = Circle((new_y, new_x), radius=8, edgecolor='white', facecolor='black', linewidth=1)
            ax.add_patch(circle)
            ax.text(new_y, new_x, str(target_i), color='white', fontsize=12, ha='center', va='center')
    else:
        x, y = target_locations
        new_x = H - (x - start_x)
        new_y = y - start_y
        # ax.scatter(new_y, new_x, s=50, color='white', marker="*", label="Target")
        # ax.text(new_y+text_offset[0], new_x+text_offset[1], '0', fontsize=3)
        circle = Circle((new_y, new_x), radius=8, edgecolor='white', facecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(new_y, new_x, str(target_i), color='white', fontsize=12, ha='center', va='center')

    # Return numpy array if no save_path provided or save and return path
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Saved cropped BEV map: {save_path}")
    # Render the figure to a numpy array
    fig.canvas.draw()
    img_result = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    return img_result

class JsonFileHandler(FileSystemEventHandler):
    def __init__(self, overall_path, scene_graph_path, img_path):
        self.overall_folder = os.path.abspath(overall_path)
        self.base_folder = os.path.abspath(scene_graph_path)
        self.img_base_folder = os.path.abspath(img_path)
    
    def on_created(self, event):
        """Trigger when a new file is created in the folder."""
        if event.is_directory:
            return

        file_path = event.src_path
        if file_path.endswith("_loc.npy"):
            print(f"New JSON file detected: {file_path}")
            self.load_and_plot(file_path)  # Start tracking file
    
    def load_and_plot(self, file_path):
        episode = os.path.dirname(file_path).split('/')[-1]
        scene = os.path.dirname(file_path).split('/')[-2]
        info = os.path.basename(file_path).split('.')[0].split('_')
        step, frontier_id = info[0], info[1]

        sub_obs_path = f"{self.base_folder}/{scene}/{episode}/{step}_{frontier_id}_observed_sub_sg.json"
        sub_pred_path = f"{self.base_folder}/{scene}/{episode}/{step}_{frontier_id}_predicted_sub_sg.json"
        obs_path = f"{self.base_folder}/{scene}/{episode}/{step}_{frontier_id}_observed_sg.json"
        map_path = f"{self.base_folder}/{scene}/{episode}/{step}_{frontier_id}_occ_map.npy"
        loc_path = f"{self.base_folder}/{scene}/{episode}/{step}_{frontier_id}_loc.npy"

        sub_original_scene_graph = load_json(sub_obs_path)
        sub_predicted_scene_graph = load_json(sub_pred_path)
        original_scene_graph = load_json(obs_path)
        predicted_scene_graph = load_json(obs_path)
        map = np.load(map_path)
        loc = np.load(loc_path)

        target_location = list(loc[0])
        agent_coordinate = list(loc[1])

        sg_save_path = f"{self.img_base_folder}/{scene}/{episode}"
        if not os.path.exists(sg_save_path):
            os.makedirs(sg_save_path)
        sg_img_save_path = f"{sg_save_path}/{step}_{frontier_id}.png"

        visualize_scenegraph_diff_with_hierarchy_3d([sub_original_scene_graph, original_scene_graph], [sub_predicted_scene_graph, predicted_scene_graph], target_location, map, agent_coordinate, edge_threshold=50, save_path=sg_img_save_path)

        sg_image = cv2.imread(sg_img_save_path)
        overall_image_path = f"{self.overall_folder}/{scene}/{episode}/a{step}.png"
        overall_image = cv2.imread(overall_image_path)
        
        sg_image = cv2.resize(sg_image, (800, 400))
        visualize_image = add_resized_image(overall_image, sg_image, (800, 40), (800, 400))
        cv2.imwrite(overall_image_path, visualize_image)
        return

def monitor_change(llm_name, experiment_name):
    # Set the path to your JSON file.
    current_dir = os.getcwd()
    overall_path = f'{current_dir}/dump/figures/{experiment_name}'
    scene_graph_path = f'{current_dir}/dump/scenegraph_{llm_name}_json/{experiment_name}'
    img_path = f'{current_dir}/dump/scenegraph_{llm_name}/{experiment_name}'
    
    # Set up the file system event handler and observer.
    event_handler = JsonFileHandler(overall_path, scene_graph_path, img_path)

    # event_handler.load_and_plot(file_path='/root/Documents/SG-Nav_new/SG-Nav/dump/scenegraph_qwen2.5_7b_json/ObjectOnly_ExploitationOnly_ObservedOnly/4ok3usBNeis.basis/20/73_0_occ_map.npy')

    observer = Observer()
    watch_dir = scene_graph_path
    observer.schedule(event_handler, path=watch_dir, recursive=True)
    observer.start()
    
    print(f"Monitoring {scene_graph_path} for changes.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping observer...")
        observer.stop()
    observer.join()
    return

def visualize_scenegraphs(sub_obs_path, sub_pred_path, obs_path, pred_path, map_path, 
                         target_x, target_y, save_path, agent_x, agent_y):
    """
    Main visualization function to be called directly or from another script.
    
    Parameters:
    -----------
    sub_obs_path : str
        Path to the observed sub-scene graph JSON file
    sub_pred_path : str
        Path to the predicted sub-scene graph JSON file
    obs_path : str
        Path to the observed complete scene graph JSON file
    pred_path : str
        Path to the predicted complete scene graph JSON file
    map_path : str
        Path to the occupancy map numpy file
    target_x, target_y : float
        Target location coordinates
    save_path : str
        Path to save the visualization image
    agent_x, agent_y : float
        Agent location coordinates
    """
    sub_original_scene_graph = load_json(sub_obs_path)
    sub_predicted_scene_graph = load_json(sub_pred_path)
    original_scene_graph = load_json(obs_path)
    predicted_scene_graph = load_json(pred_path)
    map_data = np.load(map_path)
    
    target_location = [float(target_y), float(target_x)]
    agent_location = [float(agent_x), float(agent_y)]
    
    visualize_scenegraph_diff_with_hierarchy_3d(
        [sub_original_scene_graph, original_scene_graph], 
        [sub_predicted_scene_graph, predicted_scene_graph], 
        target_location, map_data, agent_location, 
        edge_threshold=50, save_path=save_path
    )
    
    # Copy to debug folder if needed
    try:
        experiment_name = save_path.split('/')[2]
        debug_dir = f"dump/debug_figs/{experiment_name}"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)
        shutil.copy2(save_path, f"{debug_dir}/diff.png")
    except (IndexError, FileNotFoundError):
        pass


if __name__ == '__main__':
    if len(sys.argv) > 10:
        # Called with command line arguments
        sub_obs_path = sys.argv[1]
        sub_pred_path = sys.argv[2]
        obs_path = sys.argv[3]
        pred_path = sys.argv[4]
        map_path = sys.argv[5]
        target_x = sys.argv[6]
        target_y = sys.argv[7]
        save_path = sys.argv[8]
        agent_x = sys.argv[9]
        agent_y = sys.argv[10]
        
        visualize_scenegraphs(
            sub_obs_path, sub_pred_path, obs_path, pred_path, map_path,
            target_x, target_y, save_path, agent_x, agent_y
        )
    else:
        raise SystemExit("Usage: vis_scenegraph.py SUB_OBS SUB_PRED OBS PRED MAP TARGET_X TARGET_Y OUTPUT AGENT_X AGENT_Y")