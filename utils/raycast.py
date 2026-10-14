
import numpy as np 
from matplotlib import colors
unknown_rgb = colors.to_rgb('#FFFFFF')   # unexplored
free_rgb = colors.to_rgb('#E7E7E7')      # free space
obstacle_rgb = colors.to_rgb('#A2A2A2')  # obstacle

eps =0.0001

def color_distance(arr,rgb):
    return np.linalg.norm(arr-rgb,axis=-1)
def seperate_map(rgb_map):
    unknown = color_distance(rgb_map,unknown_rgb)<eps
    free = color_distance(rgb_map,free_rgb)<eps
    obstacle = color_distance(rgb_map,obstacle_rgb)<eps

    return unknown,obstacle,free

#line: shape (N x 2 x 2) batch, point, coord
def bbox(lines):
    x0 = np.floor(np.min(lines[:,0,:],axis=1)+0.5)
    y0 = np.floor(np.min(lines[:,1,:],axis=1)+0.5)
    x1 = np.ceil(np.max(lines[:,0,:],axis=1)+0.5)
    y1 = np.ceil(np.max(lines[:,1,:],axis=1)+0.5)

    return np.array((x0,x1,y0,y1),dtype=np.int32).T

#finds the coordinates of each ray
def hits(lines,bounding_boxes):
    ylines = np.flip(lines,axis=2)*1.0
    yi = lines[:,1,0]<lines[:,1,1]
    ylines[yi,:,:]= lines[yi,:,:]

    xlines = np.flip(lines,axis=2)*1.0 #jank deepcopy
    xi = lines[:,0,0]<lines[:,0,1]
    xlines[xi,:,:]= lines[xi,:,:]

    results = []

    for xline,yline,b in zip(xlines,ylines,bounding_boxes):
        Xx = np.arange(start=b[0],stop=b[1])
        slope = (xline[1,1]-xline[1,0])/(xline[0,1]-xline[0,0]) if (xline[0,1]-xline[0,0])!=0 else 0
        Yx = np.floor(xline[1,0]+slope*np.arange(b[1]-b[0])+0.5).astype(np.int32)

        Yy = np.arange(start=b[2],stop=b[3])
        slope = (yline[0,1]-yline[0,0])/(yline[1,1]-yline[1,0]) if (yline[1,1]-yline[1,0])!=0 else 0
        Xy = np.floor(yline[0,0]+slope*np.arange(b[3]-b[2])+0.5).astype(np.int32)


        xmask = (b[0]<=Xx)* (Xx<b[1]) * (b[2]<=Yx)* (Yx<b[3])
        ymask = (b[0]<=Xy) *(Xy<b[1]) * (b[2]<=Yy)* (Yy<b[3])

        Xx = Xx[xmask]
        Yx = Yx[xmask]
        Xy = Xy[ymask]
        Yy = Yy[ymask]
        #results.append((Xy,Yy))
        results.append((np.concatenate((Xx,Xy)),np.concatenate((Yx,Yy))))
    return results

#position: coordinate tuple or array
#angles: iterable of radian directions to send ray
#obstacle_map: occupancy map, 1 is obstacle
def raycast(position,angles,obstacle_map,r=20):
    rays = []
    x,y = position
    for angle in angles:
        rays.append(np.array([[x,y],
                            [x+np.cos(angle)*r,y+np.sin(angle)*r]]).T)
    rays = np.array(rays)
    bb = bbox(rays)
    #print(rays)
    pix_rays = hits(rays,bb)
    results = []
    # print(obstacle_map.shape)
    h, w = obstacle_map.shape
    for x,y in pix_rays:
        x = np.clip(x.reshape(-1),0,h-1)
        y = np.clip(y.reshape(-1),0,w-1)
        idx = obstacle_map[x,y]>0 #indices where ray intersects obstacle
        ray = np.array([x,y]).T
        #print(ray.shape)
        # intersections = ray[idx,:]
        if(np.count_nonzero(idx)>0):
            distances = np.linalg.norm(ray-np.array([[*position]]),axis=1)
            #print("hit!")
            # print(distances)
            max_distance = np.min(distances[idx])
            # print(max_distance)
            cast_ray = ray[distances<max_distance]
            #print(cast_ray)
            results.append((cast_ray[:,0],cast_ray[:,1]))
        else:
            results.append((x,y))
            #pass
    return results

def get_visible_unknown(position,n_rays,obstacle_map,unknown_map,r=60):
    antishadow = raycast(position,np.linspace(0,np.pi*2,n_rays),obstacle_map,r)
    mask = np.zeros_like(unknown_map)
    for x,y in antishadow:
        mask[x,y]=unknown_map[x,y]
    return mask

# import marshal
# import types
# from functools import partial
# def internal_func_map(pool, f, gen):
#     marshaled = marshal.dumps(f.__code__)
#     return pool.map(partial(run_func, marshaled=marshaled), gen)


# def run_func(*args, **kwargs):
#     marshaled = kwargs.pop("marshaled")
#     func = marshal.loads(marshaled)

#     restored_f = types.FunctionType(func, globals())
#     return restored_f(*args, **kwargs)
h,w = 300,300
def task(xline,yline,b,position,obstacle_map,mask,value):
    Xx = np.arange(start=b[0],stop=b[1])
    slope = (xline[1,1]-xline[1,0])/(xline[0,1]-xline[0,0]) if (xline[0,1]-xline[0,0])!=0 else 0
    Yx = np.floor(xline[1,0]+slope*np.arange(b[1]-b[0])+0.5).astype(np.int32)

    Yy = np.arange(start=b[2],stop=b[3])
    slope = (yline[0,1]-yline[0,0])/(yline[1,1]-yline[1,0]) if (yline[1,1]-yline[1,0])!=0 else 0
    Xy = np.floor(yline[0,0]+slope*np.arange(b[3]-b[2])+0.5).astype(np.int32)


    xmask = (b[0]<=Xx)* (Xx<b[1]) * (b[2]<=Yx)* (Yx<b[3])
    ymask = (b[0]<=Xy) *(Xy<b[1]) * (b[2]<=Yy)* (Yy<b[3])

    Xx = Xx[xmask]
    Yx = Yx[xmask]
    Xy = Xy[ymask]
    Yy = Yy[ymask]
    #results.append((Xy,Yy))
    x,y = np.concatenate((Xx,Xy)),np.concatenate((Yx,Yy))

    x = np.clip(x.reshape(-1),0,h-1)
    y = np.clip(y.reshape(-1),0,w-1)

    idx = obstacle_map[x,y]>0 #indices where ray intersects obstacle
    ray = np.array([x,y]).T
    #print(ray.shape)
    # intersections = ray[idx,:]
    if(np.count_nonzero(idx)>0):
        distances = np.linalg.norm(ray-np.array([[*position]]),axis=1)
        #print("hit!")
        # print(distances)
        max_distance = np.min(distances[idx])
        # print(max_distance)
        cast_ray = ray[distances<max_distance]
        #print(cast_ray)
        x,y = cast_ray[:,0],cast_ray[:,1]
    v = np.where(mask[x,y]<value,value,mask[x,y])
    mask[x,y] = v

# def task_wrapper(args):
#     return task(*args)
# import matplotlib.path as mpltPath
# from shapely.geometry import Polygon
# from skimage.draw import polygon as sk_polygon
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
def set_antishadow(position,n_rays,obstacle_map,mask,value,r=60,heading=0,theta=np.pi,mode='contour'):
    rays = []
    x,y = position
    for angle in np.linspace(heading-theta,heading+theta,n_rays):
        rays.append(np.array([[x,y],
                            [x+np.cos(angle)*r,y+np.sin(angle)*r]]).T)
    rays = np.array(rays)
    bb = bbox(rays)
    #print(rays)
    h, w = obstacle_map.shape

    ylines = np.flip(rays,axis=2)*1.0
    yi = rays[:,1,0]<rays[:,1,1]
    ylines[yi,:,:]= rays[yi,:,:]

    xlines = np.flip(rays,axis=2)*1.0 #jank deepcopy
    xi = rays[:,0,0]<rays[:,0,1]
    xlines[xi,:,:]= rays[xi,:,:]

    slopesx = (xlines[:,1,1]-xlines[:,1,0])/(xlines[:,0,1]-xlines[:,0,0])
    slopesy = (ylines[:,0,1]-ylines[:,0,0])/(ylines[:,1,1]-ylines[:,1,0])

    # if(multiprocess):
    #     from itertools import repeat
    #     with ProcessPoolExecutor() as executor:
    #         n = len(xlines)
    #         executor.map(task_wrapper,list(zip(xlines,ylines,bb,repeat(position,n),repeat(obstacle_map,n),repeat(mask,n),repeat(value,n))))
    # else:
    i=0
    boundary_points = [(position[0],position[1])]
    for xline,yline,b in zip(xlines,ylines,bb):
        Xx = np.arange(start=b[0],stop=b[1])
        slope = slopesx[i] if (xline[0,1]-xline[0,0])!=0 else 0
        Yx = np.floor(xline[1,0]+slope*np.arange(b[1]-b[0])+0.5).astype(np.int32)

        Yy = np.arange(start=b[2],stop=b[3])
        slope = slopesy[i] if (yline[1,1]-yline[1,0])!=0 else 0
        Xy = np.floor(yline[0,0]+slope*np.arange(b[3]-b[2])+0.5).astype(np.int32)


        xmask = (b[0]<=Xx)* (Xx<b[1]) * (b[2]<=Yx)* (Yx<b[3])
        ymask = (b[0]<=Xy) *(Xy<b[1]) * (b[2]<=Yy)* (Yy<b[3])

        Xx = Xx[xmask]
        Yx = Yx[xmask]
        Xy = Xy[ymask]
        Yy = Yy[ymask]
        #results.append((Xy,Yy))
        x,y = np.concatenate((Xx,Xy)),np.concatenate((Yx,Yy))

        x = np.clip(x.reshape(-1),0,h-1)
        y = np.clip(y.reshape(-1),0,w-1)

        idx = obstacle_map[x,y]>0 #indices where ray intersects obstacle
        ray = np.array([x,y]).T
        #print(ray.shape)
        # intersections = ray[idx,:]
        distances = np.linalg.norm(ray-np.array([[*position]]),axis=1)

        if mode=='fill':
            if(np.count_nonzero(idx)>0):
                max_distance = np.min(distances[idx])
                cast_ray = ray[distances<max_distance]
                #print(cast_ray)
                x,y = cast_ray[:,0],cast_ray[:,1]
            v = np.where(mask[x,y]<value,value,mask[x,y])
            mask[x,y] = v
        if mode == 'contour':
            if(np.count_nonzero(idx)>0):
                furthest_idx = np.argmin(distances[idx])
                boundary_points.append((x[idx][furthest_idx],y[idx][furthest_idx]))
            else:
                furthest_idx = np.argmax(distances)
                boundary_points.append((x[furthest_idx],y[furthest_idx]))
        i+=1
    if mode == 'contour':
        # boundary_points.insert(0,position)
        # boundary_points = np.array(boundary_points)
        # # plt.plot(boundary_points[:,1],boundary_points[:,0])
        # # path = mpltPath.Path(boundary_points)
        # polygon = Polygon(boundary_points)
        # polygon.exterior.
        # points = np.vstack((X.flatten(),Y.flatten())).T
        # # print(points)
        # interior = path.contains_points(points)
        # # print(np.count_nonzero(interior))

        # x,y = points[interior][:,0],points[interior][:,1]

        image = Image.new("1", mask.shape)
        draw = ImageDraw.Draw(image)
        draw.polygon(boundary_points, fill=1)
        # coordgrid = np.stack(np.meshgrid(np.arange(w),np.arange(h)),axis=2)
        # phi = np.ones((w,h))
        # phi[int(position[0]),int(position[1])]=0
        # distance = skfmm.distance(phi,dx=1)
        # distances = np.linalg.norm(coordgrid-position.reshape((1,1,2)),axis=2)
        v = value#*np.exp(-distance/30.)
        interior = np.array(image)*v#*np.exp(-distance/30.)
        # v = np.where(mask[interior.T]<value,value,mask)
        # mask[interior.T]=1
        mask[mask<interior.T]=v#[mask<interior.T]
        # mask = np.where(mask<interior.T,interior.T,mask)

        # boundary_points = np.array(boundary_points)
        # rr, cc = sk_polygon(boundary_points[:,0], boundary_points[:,1], shape=mask.shape)
        # v = np.where(mask[rr,cc]<value,value,mask[rr,cc])
        # mask[rr,cc] = v
    # return mask,boundary_points

def get_trajectory_exploration(trajectory,obstacle_map,unknown_map,gamma = 0.98,p= 1./(300*300),theta=1,n_rays=100,r=128):
    probabilities = np.zeros_like(unknown_map,dtype=float)
    try:
        iter(gamma)
    except TypeError:
        # not iterable
        factors = [gamma**i for i in range(len(trajectory[0]))]
    else:
        factors = gamma

    for position,heading,f in zip(*trajectory,factors):
        set_antishadow(position,n_rays,obstacle_map,probabilities,f*p,r,heading,theta,mode='contour')
    total_probability = 1-np.prod(1-probabilities)
    probabilities*=unknown_map
    return probabilities,total_probability#/total_probability

#approximates tangent vectors to a path given as sequence of points
#generated by google gemni
def approximate_tangents(path: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Approximates the tangent vector at each coordinate of a path.

    The tangent at point P_i is approximated as:
    - P_1 - P_0 for the first point (forward difference)
    - P_N-1 - P_N-2 for the last point (N-1) (backward difference)
    - P_i+1 - P_i-1 for interior points P_i (central difference)

    Args:
        path (np.ndarray): An Nx2 NumPy array of (x, y) coordinates.
        normalize (bool): If True, normalize the tangent vectors to unit length.
                          Defaults to True.

    Returns:
        np.ndarray: An Nx2 NumPy array of tangent vectors (dx, dy).
                    If a tangent vector's magnitude is close to zero (e.g., duplicate points),
                    and normalize is True, it will be returned as [0, 0].
                    Returns an empty array of shape (0,2) if input path has < 1 point.
                    Returns [[0,0]] if input path has exactly 1 point.
    """
    N = path.shape[0]

    if N == 0:
        return np.empty((0, 2), dtype=path.dtype)
    if N == 1:
        # Tangent is undefined for a single point, return [0,0] as a convention
        return np.array([[0.0, 0.0]], dtype=path.dtype)

    tangents = np.zeros_like(path, dtype=float) # Use float for potential normalization

    # First point: forward difference
    tangents[0] = path[1] - path[0]

    # Last point: backward difference
    if N > 1: # This check is technically redundant due to N==1 case above, but good for clarity
        tangents[N - 1] = path[N - 1] - path[N - 2]

    # Interior points: central difference
    # path[2:] gives elements from index 2 to end
    # path[:-2] gives elements from index 0 up to (but not including) N-2
    # So, path[2:] - path[:-2] effectively computes P[i+1] - P[i-1] for i in [1, ..., N-2]
    if N > 2:
        tangents[1:-1] = path[2:] - path[:-2]
        
    # Handle N=2 case explicitly if central diff wasn't run,
    # though the initial forward/backward handles it.
    # For N=2, tangents[0] = path[1]-path[0], tangents[1] = path[1]-path[0]
    # This is consistent with the above logic.
    return tangents

# given a sequence of points, approximate the heading change as you move between them. in radians
def get_headings(path):
    tangents = approximate_tangents(path)
    return np.arctan2(tangents[:,1],tangents[:,0])

def get_path_exploration(path,obstacle_map,unknown_map,gamma = 0.98,p= 1./(300*300),theta=1,n_rays=100,r=128):
   
    return get_trajectory_exploration((path,get_headings(path)),obstacle_map,unknown_map,gamma = gamma,p= p,theta=theta,n_rays=n_rays,r=r)
