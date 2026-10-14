import numpy as np
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_local_map(loc, map, radius=32):
    H, W = map.shape[-2:]
    y, x = loc[0], loc[1]
    # Compute bounds efficiently
    y_min, y_max = max(0, y - radius), min(H, y + radius + 1)
    x_min, x_max = max(0, x - radius), min(W, x + radius + 1)
    
    # Extract the valid sub-map slice
    sub_map = map[y_min:y_max, x_min:x_max]
    return sub_map

def parse_dict_scene_graph(scene_graph_dict):
    # Parse the scene graph dictionary to extract relevant information
    room_nodes = []
    region_nodes = []
    object_nodes = []
    for room in scene_graph_dict.get('rooms', []):
        # if room['center'] == '':
        if len(room.get('regions', [])) != 0:
            room_nodes.append({"caption": room.get("caption", ""), "corr_score": room.get("corr_score", 0.0), "center": room.get("center", [0,0])})

        for region in room.get('regions', []):
            region_nodes.append({"caption": region.get("caption", ""), "corr_score": region.get("corr_score", 0.0), "center": region.get("center", [0,0])})
            
            for object in region.get('objects', []):
                object_nodes.append({"caption": object.get("caption", ""), "corr_score": object.get("corr_score", 0.0), "center": object.get("center", [0,0])})
    
    return room_nodes, region_nodes, object_nodes

def visualize_multiple_score_maps(score_maps, labels, filename="multi_score_map.png", cmap="hot"):
    """
    Visualizes and saves multiple score maps side by side.

    Parameters:
    - score_maps: List of (H, W) NumPy arrays representing different score maps.
    - labels: List of titles for each score map.
    - filename: Path to save the output image (e.g., "multi_score_map.png").
    - cmap: Colormap to use ("hot", "jet", "viridis", etc.).
    """
    num_maps = len(score_maps)
    fig, axes = plt.subplots(1, num_maps, figsize=(5 * num_maps, 5))  # Create subplots

    for i, ax in enumerate(axes):
        im = ax.imshow(score_maps[i], cmap=cmap, interpolation="nearest")
        ax.set_title(labels[i])
        ax.set_xlabel("X (BEV)")
        ax.set_ylabel("Y (BEV)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Add colorbar to each plot

    # Save the figure before showing it
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Score maps saved as {filename}")


def distance_model(distances, sigma=10):
    p = np.exp(-distances**2 / (2*sigma**2)) # RBF
    # p = np.exp(-distances / sigma)  # Exponential decay
    return p

def compute_location_distribution(nodes, node_type, grid_size=(100, 100), sigma=10):
    """
    Compute target location distribution in BEV space efficiently using NumPy.
    
    Output:
        - bev_socre: (2, H, W), [0] is score sum, [1] is node count. 
            To get socre, divide 0th dim by 1th dim. 
    """
    H, W = grid_size

    # Create a grid of (x, y) coordinates in the image frame, z points into image, x points right, y points down
    # x_coords, y_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid_points = np.stack([x_coords, y_coords], axis=-1)  # Shape (H, W, 2)

    # Compute likelihood for all nodes in parallel
    # nodes, semantic_scores = semantic_compatibility(target, obj_corr_mat, room_corr_mat, scene_graph, node_type)
    
    # Initial uniform probability
    bev_grid = np.ones((H, W)) * 0.5
    # assume we have on observation of the all places with 0.5 prob, meaning 1 entropy
    node_count = np.ones((H, W)) # will add location filter in the future
    
    for i, node in enumerate(nodes):
        if node['center'] is not None:
            distances = np.linalg.norm(grid_points - node['center'], axis=-1)
            likelihoods = node['corr_score'] * distance_model(distances, sigma)
            likelihoods[distances > 2*sigma] = 0 # filter out the far away points
            bev_grid += likelihoods  # Sum likelihoods over all nodes
            node_count[distances <= 2*sigma] += 1 # filter out the far away points
            # node_count += 1
    # bev_grid = bev_grid / node_count if node_count > 0 else bev_grid
    return np.stack((bev_grid, node_count)) # (2, H, W)

def per_map_normalization(probability_grid):
    normalized_grid = probability_grid/probability_grid.sum()
    return normalized_grid

def compute_entropy_per_grid(normalized_grid):
    """
    Compute entropy per grid cell and sum over the entire grid.
    
    Parameters:
    - normalized_grid: (H, W) array with locally normalized probabilities.

    Returns:
    - entropy: The total entropy value.
    """
    # Compute entropy
    entropy = -normalized_grid * np.log2(normalized_grid+1e-6) - (1-normalized_grid) * np.log2(1-normalized_grid+1e-6)
    return entropy


def compute_geometric_score(spatial_candidates, unknown_map, entropy_map, radius=32):
    def normalize_nonzero_count(nonzero_count, min_val, max_val):
        """
        Normalize the non-zero element count to a range of [0, 1].

        Parameters:
        - nonzero_count (int): The raw count of non-zero elements.
        - min_val (int): The minimum possible count of non-zero elements.
        - max_val (int): The maximum possible count of non-zero elements.

        Returns:
        - float: Normalized value between 0 and 1.
        """
        if max_val == min_val:  # Avoid division by zero
            return 0.0
        
        return (nonzero_count - min_val) / (max_val - min_val)
    
    H, W = unknown_map.shape
    unknown_grids = []
    for i, loc in enumerate(spatial_candidates):
        x, y = loc[0], loc[1]
        # Compute bounds efficiently
        x_min, x_max = max(0, x - radius), min(H, x + radius + 1)
        y_min, y_max = max(0, y - radius), min(W, y + radius + 1)
        
        # Extract the valid sub-map slice
        sub_entropy_map = entropy_map[x_min:x_max, y_min:y_max]

        # Apply the mask and count non-zero elements efficiently
        unknown_grids.append(sub_entropy_map.sum())
    min_count = min(unknown_grids)
    max_count = max(unknown_grids) 
    normalized_scores = np.array([normalize_nonzero_count(count, min_count, max_count) for count in unknown_grids])
    print('Geo corr: {:.02f}-{:.02f}'.format(normalized_scores.min(), normalized_scores.max()))
    return normalized_scores