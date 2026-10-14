import cv2
import numpy as np
import torch
import skimage
from matplotlib import cm



def line_list(text, line_length=80):
    text_list = []
    for i in range(0, len(text), line_length):
        text_list.append(text[i:(i + line_length)])
    return text_list

def add_text(image: np.ndarray, text: str, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def add_dashed_line(image: np.ndarray, start_point=(0, 0), end_point=(200, 200), color=(0, 0, 0), thickness=2, dash_length=10, gap_length=5):
    """
    Draws a dashed line on the given image.

    Parameters:
        image (np.ndarray): The image on which the dashed line is to be drawn.
        start_point (tuple): Coordinates of the starting point of the line (x, y).
        end_point (tuple): Coordinates of the ending point of the line (x, y).
        color (tuple): Color of the line in BGR format (default is black).
        thickness (int): Thickness of the dashed line (default is 2).
        dash_length (int): Length of each dash (default is 10 pixels).
        gap_length (int): Length of the gap between dashes (default is 5 pixels).

    Returns:
        np.ndarray: The image with the dashed line drawn on it.
    """
    x1, y1 = start_point
    x2, y2 = end_point

    # Calculate the total distance between start and end points
    line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # Calculate the unit vector along the line
    dx = (x2 - x1) / line_length
    dy = (y2 - y1) / line_length

    # Draw dashes along the line
    current_length = 0
    while current_length < line_length:
        # Start and end points of each dash
        start_x = int(x1 + dx * current_length)
        start_y = int(y1 + dy * current_length)
        end_x = int(x1 + dx * min(current_length + dash_length, line_length))
        end_y = int(y1 + dy * min(current_length + dash_length, line_length))

        # Draw the dash
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)

        # Move to the next segment
        current_length += dash_length + gap_length

    return image

def add_line(image: np.ndarray, start_point=(0, 0), end_point=(100, 100), color=(0, 0, 0), thickness=2):
    """
    Draws a line on the given image.

    Parameters:
        image (np.ndarray): The image on which the line is to be drawn.
        start_point (tuple): Coordinates of the starting point of the line (x, y).
        end_point (tuple): Coordinates of the ending point of the line (x, y).
        color (tuple): Color of the line in BGR format (default is black).
        thickness (int): Thickness of the line (default is 2).

    Returns:
        np.ndarray: The image with the line drawn on it.
    """
    cv2.line(image, start_point, end_point, color, thickness)
    return image

def add_text_list(image: np.ndarray, text_list: list, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    for i, text in enumerate(text_list):
        position_i = (position[0], position[1] + i * 15)
        cv2.putText(image, text, position_i, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def add_rectangle(image: np.ndarray, top_left: tuple, bottom_right: tuple, color=(0, 255, 0), thickness=2):
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

def add_resized_image(base_image: np.ndarray, overlay_image: np.ndarray, position: tuple, size: tuple = None):
    '''
    Input:
        - base_image: np.ndarray, the base image to overlay on.
        - overlay_image: np.ndarray, the image to overlay.
        
    '''
    if size is None:
        resized_overlay = overlay_image
    else:
        resized_overlay = cv2.resize(overlay_image, size)

    h, w = resized_overlay.shape[:2]

    x, y = position

    if x + w > base_image.shape[1] or y + h > base_image.shape[0]:
        raise ValueError("Overlay image goes out of the bounds of the base image.")

    base_image[y:y+h, x:x+w] = resized_overlay
    return base_image

def crop_around_point(image: np.ndarray, point: tuple, size: tuple):
    img_height, img_width = image.shape[:2]
    
    crop_width, crop_height = size
    
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
    
    cropped_image = image[top:bottom, left:right]
    
    return cropped_image

def draw_agent(agent, map, pose, agent_size, color_index, vln_logger, alpha=1):
    # TODO: fix circular import
    
    agent_y_index = int((agent.mapping_module.map_size_cm/100-pose[1])*100/agent.resolution)
    agent_x_index = int(pose[0]*100/agent.resolution)
    
    # vln_logger.info(f"agent_y_index: {agent_y_index}, agent_x_index: {agent_x_index}")
    
    color_ori = map[:, agent_y_index-agent_size:agent_y_index+agent_size, agent_x_index-agent_size:agent_x_index+agent_size]
    color_new = torch.zeros_like(color_ori)
    color_new[color_index] = 1
    color_new = alpha * color_new + (1 - alpha) * color_ori
    map[:, agent_y_index-agent_size:agent_y_index+agent_size, agent_x_index-agent_size:agent_x_index+agent_size] = color_new

def draw_goal(agent, map, goal_size, color_index, vln_logger):
    skimage.morphology.disk(goal_size)
    if not agent.found_goal and agent.goal_loc is not None:
        map[:,int(agent.mapping_module.map_size_cm/5)-agent.goal_loc[0]-goal_size:int(agent.mapping_module.map_size_cm/5)-agent.goal_loc[0]+goal_size, agent.goal_loc[1]-goal_size:agent.goal_loc[1]+goal_size] = 0
        map[color_index,int(agent.mapping_module.map_size_cm/5)-agent.goal_loc[0]-goal_size:int(agent.mapping_module.map_size_cm/5)-agent.goal_loc[0]+goal_size, agent.goal_loc[1]-goal_size:agent.goal_loc[1]+goal_size] = 1
    else:
        goal_y_index =  int((agent.mapping_module.map_size_cm/200+agent.goal_gps[1])*100/agent.resolution)
        goal_x_index =  int((agent.mapping_module.map_size_cm/200+agent.goal_gps[0])*100/agent.resolution)
        
        # vln_logger.info(f"goal_y_index: {goal_y_index}, goal_x_index: {goal_x_index}")
        
        # free space
        map[:, goal_y_index-goal_size:goal_y_index+goal_size, goal_x_index-goal_size:goal_x_index+goal_size] = 0
        
        # goal occupied space
        map[color_index, goal_y_index-goal_size:goal_y_index+goal_size, goal_x_index-goal_size:goal_x_index+goal_size] = 1

def draw_frontier_map(map_tensor, frontier_locations_16, scores, grid_size=2):
    """
    Draws a frontier map with color-coded frontier locations.

    Args:
        map_tensor (np.ndarray): [h, w, 3] tensor representing the map.
        frontier_locations_16 (np.ndarray): [N, 2] array of frontier locations (y, x).
        scores (np.ndarray): [N] array of scores for the frontier locations.
    """
    # Normalize scores to the range [0, 1]
    # normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    normalized_scores = scores

    # Create a colormap 
    colormap = cm.get_cmap('rainbow')

    # Make a copy of the map tensor to modify
    frontier_map = torch.flip(map_tensor, dims=[0])

    # Overlay the frontier locations
    for i, (y, x) in enumerate(frontier_locations_16):
        if 0 <= y < frontier_map.shape[0] and 0 <= x < frontier_map.shape[1]:
            # Get the color from the colormap based on the normalized score
            color = colormap(normalized_scores[i])
            frontier_map[y-grid_size:y+grid_size, x-grid_size:x+grid_size] = torch.tensor(color[:3]).double()  # Extract RGB values
    return torch.flip(frontier_map, dims=[0])

def draw_frontier_score(map_tensor, frontiers, grid_size=2):
    """
    Draws a frontier map with color-coded frontier locations.

    Args:
        map_tensor (np.ndarray): [h, w, 3] tensor representing the map.
        frontier_locations_16 (np.ndarray): [N, 2] array of frontier locations (y, x).
        scores (np.ndarray): [N] array of scores for the frontier locations.
    """
    # Normalize scores to the range [0, 1]
    # normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # Create a colormap 
    frontier_map = map_tensor
    H, W = frontier_map.shape[:2]
    if 'spatial_locs' not in frontiers:
        return frontier_map
    frontier_locations_16 = frontiers['spatial_locs']
    exploit_scores = frontiers['exploitation_scores']
    explore_scores = frontiers['exploration_scores']
    distance_scores = frontiers['distance_scores']
    # Overlay the frontier locations
    frontier_map = cv2.UMat(frontier_map)
    for i, (y, x) in enumerate(frontier_locations_16):
        if 0 <= y < H and 0 <= x < W:
            frontier_map = add_text(frontier_map, '{:.02f},{:.02f},{:.02f}'.format(exploit_scores[i], explore_scores[i], distance_scores[i]), (x, H-y), font_scale=0.3, thickness=1)
    frontier_map = frontier_map.get()
    return frontier_map

def draw_scenegraph(map, scenegraph):
    """
    Visualizes the scenegraph by drawing connections, captions, and nodes
    for rooms, groups, and objects, with a dynamically sized white background.

    Parameters:
        scenegraph: The scenegraph containing room, group, and object nodes.

    Returns:
        np.ndarray: The image with the visualized scenegraph.
    """
    img_height, img_width = map.shape

    # Determine the number of room nodes and group nodes to set the background size
    # valid_rooms = [room_node for room_node in scenegraph.room_nodes if len(room_node.group_nodes) > 0]
    # num_rooms = len(valid_rooms)
    num_rooms = len(scenegraph.room_nodes)

    # Dynamically set the background size
    white_background_height = 120
    total_height = img_height + white_background_height

    # Create a white background
    extended_image = np.ones((total_height, img_width, 3), dtype=np.uint8) * 255  # White background

    # Calculate positions for room node centers (above the main area)
    room_y_offset = white_background_height // 2  # Center of the white background
    room_x_step = img_width // (num_rooms + 1)  # Equally spaced along the width

    # Assign centers to room nodes
    for idx, room_node in enumerate(scenegraph.room_nodes):
        room_center_x = (idx + 1) * room_x_step
        room_center_y = room_y_offset
        room_node.center = (room_center_x, room_center_y)  # Set center for the room node

    # Colors for visualization
    room_color = (225, 170, 170)  # Dark Pink (BGR)
    group_color = (190, 120, 170)  # Bright Pink (BGR)
    object_color = (210, 170, 100)  # Light Blue (BGR)

    # Font scales for captions
    room_font_scale = 0.6
    group_font_scale = 0.5
    object_font_scale = 0.6

    # Line thickness for clarity
    dashed_line_thickness = 2
    solid_line_thickness = 2

    # Visualize the scenegraph
    for room_node in scenegraph.room_nodes:
        room_caption = room_node.caption
        room_center = room_node.center

        # Draw a circle at the room center
        cv2.circle(extended_image, room_center, radius=12, color=room_color, thickness=-1)  # Filled circle

        # Draw room caption
        add_text(extended_image, room_caption, position=(room_center[0] - 40, room_center[1] - 20),
                 font_scale=room_font_scale, color=room_color)

        # Visualize group nodes within the room
        for group_node in room_node.group_nodes:
            group_center = (int(group_node.center[0]), int(group_node.center[1]) + white_background_height)
            group_caption = group_node.caption

            # Draw a dashed line between room center and group center
            add_dashed_line(extended_image, start_point=room_center, end_point=(group_center[0], group_center[1] - 200),
                            color=group_color, thickness=dashed_line_thickness, dash_length=15, gap_length=10)

            # Draw a circle at the group center
            cv2.circle(extended_image, (group_center[0], group_center[1] - 200), radius=10, color=group_color, thickness=-1)  # Filled circle

            # Draw group caption
            add_text(extended_image, group_caption, position=(group_center[0] - 30, group_center[1] - 220),
                     font_scale=group_font_scale, color=group_color)

            # Visualize object nodes within the group
            for object_node in group_node.nodes:
                object_center = (int(object_node.center[0]), int(object_node.center[1]) + white_background_height)
                object_caption = object_node.caption

                # Draw a solid line between group center and object center
                add_line(extended_image, start_point=(group_center[0], group_center[1] - 200), end_point=object_center,
                         color=object_color, thickness=solid_line_thickness)

                # Draw a circle at the object center
                cv2.circle(extended_image, object_center, radius=8, color=object_color, thickness=-1)  # Filled circle

                # Draw object caption
                add_text(extended_image, object_caption, position=(object_center[0] - 30, object_center[1] - 20),
                         font_scale=object_font_scale, color=object_color)

    # Add a legend to the image
    extended_image = add_legend(extended_image, room_color, group_color, object_color)

    return extended_image


def add_legend(image, room_color, group_color, object_color):
    """
    Adds a color legend to the visualization using circles.

    Parameters:
        image (np.ndarray): The image where the legend will be added.
        room_color (tuple): The color for Room Level.
        group_color (tuple): The color for Group Level.
        object_color (tuple): The color for Object Level.

    Returns:
        np.ndarray: The image with the added legend.
    """
    legend_x, legend_y = 20, 700  # Bottom-left corner of the legend
    circle_radius = 10
    spacing = 40
    font_scale = 0.6
    text_offset = 15  # Offset for text from the circle

    # Draw Room Level Legend
    cv2.circle(image, (legend_x + circle_radius, legend_y + circle_radius), circle_radius, room_color, -1)  # Filled circle
    add_text(image, "Room Level", position=(legend_x + 2 * circle_radius + 10, legend_y + circle_radius + text_offset),
             font_scale=font_scale, color=(0, 0, 0))

    # Draw Group Level Legend
    cv2.circle(image, (legend_x + circle_radius, legend_y + spacing + circle_radius), circle_radius, group_color, -1)
    add_text(image, "Group Level", position=(legend_x + 2 * circle_radius + 10, legend_y + spacing + circle_radius + text_offset),
             font_scale=font_scale, color=(0, 0, 0))

    # Draw Object Level Legend
    cv2.circle(image, (legend_x + circle_radius, legend_y + 2 * spacing + circle_radius), circle_radius, object_color, -1)
    add_text(image, "Object Level", position=(legend_x + 2 * circle_radius + 10, legend_y + 2 * spacing + circle_radius + text_offset),
             font_scale=font_scale, color=(0, 0, 0))

    return image

def draw_bbox_with_label(rgb, xyxy, captions, conf=None, color=(255,0,0)):
    '''
    Draw bounding boxes with labels on the RGB image
    
    Input:
        - rgb: np.array (H,W,3)
        - xyxy: np.array (n,4)
        - captions: list of strings
        - conf(optional): np.array (n,)
        - color: tuple of 3 integers 0~255 (b,g,r)
        
    Ouput:
        - rgb: np.array (H,W,3)
    '''
    
    # preprocess the rgb image
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8)

    if rgb.shape[-1] == 3:  
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    for i, (box, label) in enumerate(zip(xyxy, captions)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
        text = f"{label}"
        if conf is not None:
            text += f" ({conf[i]:.2f})"
        if color == (0,0,255):
            y1 += 20 # red is ground truth, give it another color
        cv2.putText(rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return rgb