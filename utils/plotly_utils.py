# plot the camera

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt



def plot_camera(fig, rotation_matrix, translation_vector, focal_length, legend_name, show_legend):
    # Define the camera center
    center = translation_vector

    # Calculate the camera frame size based on the focal length
    frame_width = 2 * focal_length
    frame_height = 2 * focal_length

    # Calculate the corner points of the camera frame
    corner_points = [
        center +
        rotation_matrix @ np.array([frame_width / 2,
                                   frame_height / 2, focal_length]),
        center +
        rotation_matrix @ np.array([-frame_width / 2,
                                   frame_height / 2, focal_length]),
        center +
        rotation_matrix @ np.array([-frame_width /
                                   2, -frame_height / 2, focal_length]),
        center +
        rotation_matrix @ np.array([frame_width /
                                   2, -frame_height / 2, focal_length])
    ]

    # Add the camera frame as lines
    fig.add_trace(go.Scatter3d(
        x=[corner_points[0][0], corner_points[1][0], corner_points[2]
            [0], corner_points[3][0], corner_points[0][0]],
        y=[corner_points[0][1], corner_points[1][1], corner_points[2]
            [1], corner_points[3][1], corner_points[0][1]],
        z=[corner_points[0][2], corner_points[1][2], corner_points[2]
            [2], corner_points[3][2], corner_points[0][2]],
        mode='lines',
        line=dict(color='blue', width=2),
        legendgroup=legend_name,
        name=legend_name,
        showlegend=show_legend,
        hoverinfo='text',
        text=legend_name
    ))

    # Connect the corners to the camera center
    for point in corner_points:
        fig.add_trace(go.Scatter3d(
            x=[point[0], center[0]],
            y=[point[1], center[1]],
            z=[point[2], center[2]],
            mode='lines',
            line=dict(color='blue', width=2),
            legendgroup=legend_name,
            name='',
            showlegend=False,
            hoverinfo='text',
            text=legend_name
        ))


def plot_ellipsoid(fig, center, covariance, legend_name, color=None, text='', steps=5, opacity=0.5):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scales = np.sqrt(eigenvalues)
    u = np.linspace(0, 2 * np.pi, steps)
    v = np.linspace(0, np.pi, steps)
    x = scales[0] * np.outer(np.cos(u), np.sin(v))
    y = scales[1] * np.outer(np.sin(u), np.sin(v))
    z = scales[2] * np.outer(np.ones_like(u), np.cos(v))
    ellipsoid_points = np.dot(eigenvectors, np.array(
        [x.flatten(), y.flatten(), z.flatten()]))
    x_ellipsoid = ellipsoid_points[0].reshape(x.shape)+center[0]
    y_ellipsoid = ellipsoid_points[1].reshape(y.shape)+center[1]
    z_ellipsoid = ellipsoid_points[2].reshape(z.shape)+center[2]

    if color is not None:
        colorscale = [[0, color], [1, color]]
    else:
        colorscale = None

    fig.add_trace(go.Surface(
        x=x_ellipsoid,
        y=y_ellipsoid,
        z=z_ellipsoid,
        opacity=opacity,
        colorscale=colorscale,
        showscale=False,
        name=legend_name,
        hoverinfo='text',
        text=text if text else legend_name
    ))


def plot_point(fig, point, legend_name, color=None, text=''):
    fig.add_trace(go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        mode='markers',
        marker=dict(size=5, color=color),
        name=legend_name,
        hoverinfo='text',
        text=text if text is not None else legend_name
    ))


def plot_points(fig, points, legend_name, color=None, text='', marker=None, mode='markers'):
    if marker is None:
        marker = dict(size=5, color=color)
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode=mode,
        marker=marker,
        name=legend_name,
        hoverinfo='x+y+z+name+text',
        text=text if text is not None else legend_name
    ))


def quat_to_dir(q):
    qw, qx, qy, qz = q
    # Compute the direction of the x-axis after applying the quaternion rotation
    x_dir = 1 - 2*(qy**2 + qz**2)
    y_dir = 2*(qx*qy + qw*qz)
    z_dir = 2*(qx*qz - qw*qy)
    return np.array([x_dir, y_dir, z_dir])


def plot_traj(fig, dataset, legend_name, color=None, text='', marker=None, n_start=0, n_end=None):
    traj = []
    forward = np.array([0, 1, 0])
    if n_end is None:
        n_end = dataset.get_total_number()
    pose0_inv = None
    dataset.reset()
    for i in dataset:
        if pose0_inv is None:
            pose0_inv = dataset.read_current_ground_truth().inv()
        pose = dataset.read_current_ground_truth()
        # pose = pose0_inv@dataset.read_current_ground_truth()
        traj.append(np.concatenate([pose.t, pose.R@forward]))
    traj = np.array(traj)
    plot_points(fig, traj[:, :3], legend_name,
                color, text, marker, mode='lines')
    # add a start point
    plot_point(fig, traj[0, :3], legend_name+'_start', color, text)
    # fig.add_trace(go.Cone(
    #     x=traj[:,0],
    #     y=traj[:,1],
    #     z=traj[:,2],
    #     u=traj[:,3],
    #     v=traj[:,4],
    #     w=traj[:,5],
    #     sizemode="absolute",
    #     sizeref=300,
    #     anchor="tip",
    #     showscale=False,
    #     text=text if text else legend_name
    # ))


def plot_ray_from_camera(fig, camera_pose, focal_length, point, legend_name):
    # Extract camera center and rotation matrix from the camera pose
    center = camera_pose[:3]

    # Calculate the direction vector from the camera center to the point
    direction = point - center

    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # Calculate the end point of the ray using the focal length
    ray_end = center + direction * focal_length

    # Add the ray as a line segment
    fig.add_trace(go.Scatter3d(
        x=[center[0], ray_end[0]],
        y=[center[1], ray_end[1]],
        z=[center[2], ray_end[2]],
        mode='lines',
        line=dict(color='green', width=2),
        legendgroup=legend_name,
        name=legend_name
    ))


def plot_ts_pc(fig, pc, legend_name, marker_line_width=0, marker_line_color='black', marker_size=5):
    points = pc.coords
    colors = pc.select_channels(['R', 'G', 'B']).clip(0, 1)
    colors[:, :3] = colors[:, :3] * 255.0
    colors = [f'rgb({r:.3f},{g:.3f},{b:.3f})' for r, g, b in colors]
    text = [f'({x},{y},{z})' for x, y, z in points]
    marker = dict(size=marker_size, color=colors, line=dict(
        width=marker_line_width, color=marker_line_color))
    plot_points(fig, points, legend_name, colors, text, marker)


def plot_o3d_pc(fig, pc, legend_name, marker_line_width=0, marker_line_color='black', marker_size=5):
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)
    colors = [f'rgb({r:.3f},{g:.3f},{b:.3f})' for r, g, b in colors]
    text = [f'({x},{y},{z})' for x, y, z in points]
    marker = dict(size=marker_size, color=colors, line=dict(
        width=marker_line_width, color=marker_line_color))
    plot_points(fig, points, legend_name, colors, text, marker)


def show_image(img, hoverinfo='x+y+z', name=''):
    fig = px.imshow(img)
    fig.update_traces(hoverinfo=hoverinfo, name=name)
    fig.show()

# def save_image(path, img, hoverinfo='x+y+z', name='', scale=8):
#     if img.max() > 1:
#         img = img / 255.0
#     fig = px.imshow(img)
#     fig.update_traces(hoverinfo=hoverinfo, name=name)
#     fig.update_layout(
#         coloraxis_showscale=False,
#         margin=dict(l=0, r=0, t=0, b=0)
#     )
#     fig.update_xaxes(visible=False)
#     fig.update_yaxes(visible=False)

#     # Save high-res image
#     fig.write_image(path, scale=scale)

def save_image(path, img, scale=10):
    # Normalize if values exceed 1 (assuming uint8 image)
    if img.max() > 1:
        img = img / 255.0

    # Set figure size based on image shape and scale
    height, width = img.shape[:2]
    dpi = 100 * scale  # Scale factor
    figsize = (width / dpi, height / dpi)

    # Plot and save
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def show_depth(depth, max_depth=30, hoverinfo='x+y+z', name=''):
    clipped_depth = depth.clip(0, max_depth)
    fig = px.imshow(clipped_depth, color_continuous_scale='gray')
    fig.update_traces(hoverinfo=hoverinfo, name=name)
    fig.show()


def create_fig(width=None, height=None, img=None):
    fig = go.Figure()
    fig.update_layout(scene_aspectmode='cube')
    if img is not None:
        fig.add_trace(go.Image(z=img))
        height, width = img.shape[:2]
    else:
        if width is None:
            width = 1000
        if height is None:
            height = 1000
    fig.update_layout(width=width, height=height)
    return fig


def plot_torch_pc(fig, pc, color=None, legend_name=None, marker=None):
    """points: (3, N)"""
    points = pc.squeeze().transpose(1, 0).detach().cpu().numpy()
    text = [f'({x:.2f},{y:.2f},{z:.2f})' for x, y, z in points]
    colors = None
    if color is not None:
        colors = color.squeeze().transpose(1, 0).detach().cpu().numpy()
        if marker is not None:
            marker['color'] = colors
        text = [f'pos=({x:.2f},{y:.2f},{z:.2f}), c=({r:.2f},{g:.2f},{b:.2f})' for (
            x, y, z), (r, g, b) in zip(points, colors)]
    plot_points(fig, points, legend_name, colors, marker=marker, text=text)

def plot_heatmap(fig, data, legend_name, colorscale='Viridis', x=None, y=None):
    fig.add_trace(go.Heatmap(z=data, x=x, y=y, name=legend_name, colorscale=colorscale))


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

def plot_region(fig, regions, map_size, show_objects=False, min_region_size=3):
    """
    Plotly version of the plot_region function to visualize regions on a 2D map.
    
    Args:
        fig: A plotly figure object where regions will be plotted
        regions: List of region data with grid_bbox, center, id, caption, and objects
        map_size: Size of the map (assuming square map)
        show_objects: Whether to show objects within regions
        
    Returns:
        Updated plotly figure with regions plotted
    """
    import plotly.graph_objects as go
    import matplotlib as mpl
    import numpy as np
    
    # Create a colormap for the regions
    viridis = mpl.colormaps['gist_rainbow'].resampled(len(regions))
    colors = (viridis(range(len(regions)))[:, :3] * 255).astype(int)
    
    for i, region in enumerate(regions):
        # Get color for this region
        color = f'rgb({colors[i][0]}, {colors[i][1]}, {colors[i][2]})'
        
        # Plot objects if requested
        if show_objects:
            obj_centers = []
            obj_captions = []
            for obj in region.get('objects', []):
                obj_centers.append([obj['center'][1], map_size-obj['center'][0]])
                obj_captions.append(obj.get('caption', 'unknown'))
            obj_centers = np.array(obj_centers)
            if len(obj_centers) > 0:
                fig.add_trace(go.Scatter(
                    x=obj_centers[:,0],
                    y=obj_centers[:,1],
                    mode='markers',
                    marker=dict(size=5, color='rgba(0,0,0,0)', line=dict(width=2, color='red')),
                    showlegend=False,
                    hoverinfo='x+y+text',
                    text=obj_captions
                ))
        
        if len(region['objects']) < min_region_size:
            continue
        
        # Plot the region boundary
        if isinstance(region['grid_bbox'], list):
            pts = np.array(region['grid_bbox'])
        else:
            pts = region['grid_bbox']
        pts = pts.reshape((-1, 2)).copy()
        pts[:, 0], pts[:, 1] = pts[:, 1], map_size - pts[:, 0]
        
        # Close the polygon by adding the first point at the end
        x_values = list(pts[:, 0])
        y_values = list(pts[:, 1])
        x_values.append(x_values[0])
        y_values.append(y_values[0])
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color=color, width=2),
            fill=None,
            showlegend=False,
            hoverinfo='x+y+text',
            text=f"Region {region['id']}: {region['caption']}"
        ))
        
        if isinstance(region['caption'], dict):
            captions, confidences = get_sorted_key_value_lists(region['caption'])
            region_caption = captions[0]
        else:
            region_caption = region['caption']
        
        if region.get("predicted", False):
            text_color = 'blue'
        else:
            text_color = 'black'

        # Plot the region center
        region_center = np.array([region['center'][1], map_size-region['center'][0]]).astype(np.int32)
        fig.add_trace(go.Scatter(
            x=[region_center[0]],
            y=[region_center[1]],
            mode='markers+text',
            marker=dict(size=10, color='yellow'),
            text=f"({region['id']}: {region_caption})",
            textposition="top center",
            textfont=dict(size=10, color=text_color),
            showlegend=False,
            hoverinfo='x+y+text',
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)'),
            hovertext=f"Region {region['id']}: {region['caption']}"
        ))
    
    return fig


def plot_matches(fig_k, fig_q, matches, reversed=False, map_size=480):
    # img = np.concatenate([fig_k.data[0].z, fig_q.data[0].z], axis=1)
    # merge two plots into one, put them side by side
    fig = go.Figure(fig_k)
    fig_q = go.Figure(fig_q)
    # shift all coordinates of fig_q to the right
    for data in fig_q.data:
        if data.type == 'scatter':
            data.x = np.array(data.x)+map_size
        elif data.type == 'image':
            data.x0 = map_size
        fig.add_trace(data)


    for match in matches:
        k_center = np.array([match['k']['center'][1], map_size-match['k']['center'][0]]).astype(np.int32)
        q_center = np.array([match['q']['center'][1], map_size-match['q']['center'][0]]).astype(np.int32)
        if reversed:
            k_center += np.array([map_size, 0])
        else:
            q_center += np.array([map_size, 0])
        fig.add_trace(go.Scatter(
            x=[k_center[0], q_center[0]], 
            y=[k_center[1], q_center[1]],
            mode='markers+lines',
            marker=dict(size=10, color='red', line=dict(width=2, color='yellow')),
            showlegend=False,
            hoverinfo='x+y+text',
            text=f"i:{match['k']['caption']}, j:{match['q']['caption']}, corr_score: {match['corr_score']:.2f}, sim_scores: {match['sim_scores']}",
        ))
        center_pos = (k_center+q_center)//2
        fig.add_annotation(
            x=center_pos[0], y=center_pos[1],
            text=f"{match['corr_score']:.2f}",
            showarrow=False,
            font=dict(size=12, color="yellow"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="yellow",
            borderwidth=1,
            borderpad=2,
            hovertext='x+y+text',
        )

    # Update layout for better visualization
    fig.update_layout(
        height=fig_k.layout.height*1.5,
        width=map_size * 2*1.5,
        plot_bgcolor='black',
        showlegend=False
    )
    return fig


def convert_image_to_plotly(img):
    """
    Convert an OpenCV/Numpy image to a plotly figure.
    
    Args:
        img: RGB or BGR numpy array representing an image
        
    Returns:
        A plotly figure object with the image
    """
    import plotly.express as px
    import numpy as np
    
    # Ensure image is RGB if it's BGR
    if img.ndim == 3 and img.shape[2] == 3:
        img_rgb = img[..., ::-1] if img.dtype != np.uint8 else img
    else:
        img_rgb = img
    
    # Create a plotly figure with the image
    fig = px.imshow(img_rgb, origin='lower')
    
    # Update layout for better visualization
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor='black',
        paper_bgcolor='black',
        coloraxis_showscale=False
    )
    
    return fig
