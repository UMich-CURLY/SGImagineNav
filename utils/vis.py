import dataclasses
import cv2

import numpy as np
import matplotlib.pyplot as plt

import supervision as sv
from supervision.draw.color import Color, ColorPalette

# Copied from https://github.com/concept-graphs/concept-graphs/     
def vis_result_fast(
    image: np.ndarray, 
    detections: sv.Detections, 
    classes: list[str], 
    color: Color=ColorPalette.DEFAULT, 
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> np.ndarray:
    '''
    Annotate the image with the detection results. 
    This is fast but of the same resolution of the input image, thus can be blurry. 
    '''
    # annotate image with detections
    # sv.BoundingBoxAnnotator
    box_annotator = sv.BoxAnnotator(
        color = color,
        thickness=1,
        # text_scale=0.3,
        # text_thickness=1,
        # text_padding=2,
    )
    sv.MaskAnnotator(
        color = color
    )
    label_annotator = sv.LabelAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _
        in detections]
    
    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
        
    # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections) # somhow draw mask is very slow for gt detections
    annotated_image = image.copy()
    
    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections) #, labels=labels)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image

def init_vis_image(goal_name, action = 0):
    # vis_image = np.ones((537, 1165, 3)).astype(np.uint8) * 255
    vis_image = np.ones((537 * 3 + 100, 1165 * 2, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Observations" 
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Find {}  Action {}".format(goal_name, str(action))
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color
    
    #lower right box
    # vis_image[580-1, 670:1150] = color # top
    # vis_image[530*2, 670:1150] = color # bottom
    # vis_image[580:530*2, 669] = color # left
    # vis_image[580:530*2, 1150] = color # right
    
    vis_image = draw_box(vis_image, (1200, 50), 480, 480, color)


    #     # draw legend
    #     lx, ly, _ = legend.shape
    #     vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image

def remove_image_border(vis_image: np.ndarray, padding: int = 10, black_bg: bool = False) -> np.ndarray:
    """
    Remove white border from the image.
    @param vis_image: The image to remove the white border from.
    @param padding: The padding to add to the image.
    @param bg: The background color to use for the image. (0 for black, 255 for white)
    """
    image_h, image_w = vis_image.shape[:2]
    gray = cv2.cvtColor(vis_image, cv2.COLOR_BGR2GRAY)
    if black_bg:
        _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(255-binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x += w//2
        y += h//2
        # fit the ratio of the original image
        ratio = image_w / image_h
        if w / h > ratio:
            h = int(w / ratio)
        else:
            w = int(h * ratio)
        x = int(x - w / 2)
        y = int(y - h / 2)
        # Add a small padding
        x = max(0, x)
        y = max(0, y)
        w = min(image_w - x, w)
        h = min(image_h - y, h)
        vis_image = vis_image[y:y+h, x:x+w]
        # Resize back to original dimensions
        vis_image = cv2.resize(vis_image, (image_w, image_h), interpolation=cv2.INTER_AREA)
    return vis_image

def add_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float = 1.0,
    color: tuple[int, int, int] = (20, 20, 20),
    thickness: int = 2,
    vertical_align: str = 'top',
    horizontal_align: str = 'left',
) -> np.ndarray:
    """
    Add text to the image. Return image if image is not None, otherwise return text size.
    @param color: (B, G, R) format
    @param vertical_align: 'top', 'center', 'bottom'
    @param horizontal_align: 'left', 'center', 'right'
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize_w, textsize_h = cv2.getTextSize(text, font, font_scale, thickness)[0]

    if horizontal_align == 'center':
        position = (position[0] - textsize_w // 2, position[1])
    elif horizontal_align == 'right':
        position = (position[0] - textsize_w, position[1])
    if vertical_align == 'center':
        position = (position[0], position[1] + textsize_h // 2)
    elif vertical_align == 'top':
        position = (position[0], position[1] + textsize_h)

    if image is not None:
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        return image, (textsize_w, textsize_h)
    else:
        return (textsize_w, textsize_h)

def draw_box(image, start, h, w, color, thickness=1):
    x1, y1 = start
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(image, (x1-1, y1-1), (x2+1, y2+1), color, thickness)
    return image

def resize_with_height(image, height):
    """
    Calculate new shape of height-resized image
    """
    scale = height / image.shape[0]
    new_width = int(image.shape[1] * scale)
    return [new_width, height]

def add_img(vis_image, img, start):
    x1, y1 = start
    h, w = img.shape[:2]
    x2, y2 = x1 + w, y1 + h
    x2 = min(x2, vis_image.shape[1])
    y2 = min(y2, vis_image.shape[0])
    vis_image[y1:y2, x1:x2] = img[:y2, :x2]
    return vis_image

# def add_text(vis_image, text, start, color):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 1
#     # color = (20, 20, 20)  # BGR
#     thickness = 2

#     textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
#     textX = start[0]
#     textY = start[1]
#     vis_image = cv2.putText(vis_image, text, (textX, textY),
#                             font, fontScale, color, thickness,
#                             cv2.LINE_AA)
#     return vis_image
        
def draw_text_with_circle(img, center, text):
    TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.35
    TEXT_THICKNESS = 1
    
    cv2.circle(img, center, 10, (0,0,0), TEXT_THICKNESS)

    text_size, _ = cv2.getTextSize(text, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (int(center[0] - text_size[0] / 2), int(center[1] - text_size[1] / 2))

    cv2.putText(img, text, text_origin, TEXT_FACE, TEXT_SCALE, (0,0,0), TEXT_THICKNESS, cv2.LINE_AA, bottomLeftOrigin=True)
    
    return img

def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat

def get_pyplot_as_numpy_img(img, cmap='plasma', figsize=(4.8, 4.8), origin='lower'):
    '''
    Given image you want to plot, return a numpy array of the plot. So you can 
    manipulate this image later, or add to a canvas for visualization.
    
    NOTE: pixel size = figsize * 100
    '''
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img, cmap=cmap, origin=origin)
    if origin == 'upper':
        ax.yaxis.set_inverted(True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.canvas.draw()
    # buf = fig.canvas.tostring_rgb() # consider using buffer_rgba() instead in 3.10
    buf = fig.canvas.buffer_rgba()
    ncols, nrows = fig.canvas.get_width_height()
    img_buffer = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)[..., :3]
    final_img = cv2.cvtColor(img_buffer, cv2.COLOR_RGB2BGR)
    # cbar.ax.tick_params(labelsize=5)
    plt.close()
    return final_img

def pad_frames_to_same_size(vis_frames):
    new_vis_Frames = []
    max_size = (0, 0)
    for frame in vis_frames:
        max_size = (max(max_size[0], frame.shape[0]), max(max_size[1], frame.shape[1]))
    for i, frame in enumerate(vis_frames):
        # create empty canvas
        canvas = np.ones((max_size[0], max_size[1], 3), dtype=np.uint8) * 255
        # add frame to canvas
        h_gap = (max_size[0] - frame.shape[0])//2
        w_gap = (max_size[1] - frame.shape[1])//2
        canvas[h_gap:h_gap+frame.shape[0], w_gap:w_gap+frame.shape[1]] = frame
        new_vis_Frames.append(canvas) 
    
    return new_vis_Frames