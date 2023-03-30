import cv2
import numpy as np
import torch.nn.functional as F
from matplotlib.lines import Line2D
import torch
from utils import corners3d, rotate, perspective, bbox_corners
import matplotlib.pyplot as plt
def draw_projected_box3d(image, corners3d, color, thickness=2):
    """Draw 3d bounding box in image
    
    Args:
        image (np.array): RGB image
        corners3d (np.array): (8,3) array of vertices (in image plane) for the 3d box in following order:
            4 -------- 7
           /|         /|
          5 -------- 6 .
          | |        | |
          . 0 -------- 3
          |/         |/
          1 -------- 2
        color (tupple): color of the bounding box
        thickness (int): thickness of the box
    Returns:
        image (np.array): output image
    """

    corners3d = corners3d.astype(np.int32)
    rect = np.copy(image)
    cv2.rectangle(rect, (corners3d[5, 0], corners3d[5, 1]), (corners3d[2, 0], corners3d[2, 1]), color, -1)
    image = cv2.addWeighted(image, 0.8, rect, 0.2, 0)

    for k in range(4):
        i, j = k, (k + 1) % 4
        cv2.line(image, corners3d[i], corners3d[j], color, thickness, lineType=cv2.LINE_AA)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, corners3d[i], corners3d[j], color, thickness, lineType=cv2.LINE_AA)
        i, j = k, k + 4
        cv2.line(image, corners3d[i], corners3d[j], color, thickness, lineType=cv2.LINE_AA)

    return image

def choose_color1(name):
    colors = {
        'Car': (0, 0, 255),
        'Van': (255, 192, 203),
        'Truck': (255, 255, 0),
        'Pedestrian': (128, 0, 128),
        'Cyclist': (0, 255, 0)
    }
    return colors.get(name, (255, 255, 255))  # default color: white

def choose_color2(name):
    colors = {
        'Car': 'tab:blue',
        'Van': 'tab:orange',
        'Truck': 'tab:pink',
        'Pedestrian': 'tab:purple',
        'Cyclist': 'tab:green'
    }
    return colors.get(name, 'tab:gray')  # default color: gray


def draw_3d_boxes(img, objects, calib):
    """Draw 3D bounding box with each object in image
    Args:
        image (np.array): RGB image
        objects (list of nametupled): list of object in image
        calib (torch.tensor): intrinsic matrix with shape (3, 4)
    Returns:
        image (np.array): output image with 3D bounding box
    """
    img = np.array(img)  # convert to NumPy array

    for object in objects:
        if object.classname in ['Car', 'Van', 'Truck', 'Pedestrian','Cyclist']: 
            if object.score is not None: score = object.score
            if isinstance(score, torch.Tensor): score = round(score.item(), 2)
            name = object.classname
            color = choose_color1(name)
            corners_3d = corners3d(object, calib)

            # Draw 3d bounding box
            img = draw_projected_box3d(img, corners_3d, color)

            # Find location for label
            points = [(int(corners_3d[i, 0]), int(corners_3d[i, 1])) for i in range(4)]
            min_x = min(point[0] for point in points)
            max_y = max(point[1] for point in points)
          
            # Show label
            label_size, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            score_size, baseline2 = cv2.getTextSize(str(score) + " ", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (min_x, max_y+10), (min_x+label_size[0]+score_size[0], max_y-label_size[1]+10), (0,0,0), cv2.FILLED)
            img = cv2.putText(img, name + " " + str(score), (min_x, max_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            #center = np.mean(corners_3d, axis = 0).astype(np.int32)
            #img = cv2.circle(img, (center[0], center[1]), 5, color, -1)
    return img

def vis_score(image, calib, objects, grid, cmap='binary', ax=None):
    """Visualize object labels and scores on a grid.
    Args:
        image (PIL image): input image
        calib (torch.tensor): calibration matrix
        objects (list): list of object labels
        grid (torch.tensor): grid to plot scores on
        cmap (str): color map to use for scores
        ax (matplotlib axis): optional axis object to plot on
    Returns:
        ax (matplotlib axis): axis object with bird eye view 
    """
    score = torch.zeros(grid.shape[0]-1, grid.shape[1]-1)  # adjust size to match X and Y dimensions of grid
    grid = grid.cpu().detach().numpy()

    # Create a new axis if one is not provided
    if ax is None:
        _, ax = plt.subplots()

    # Plot scores
    ax.clear()
    ax.pcolormesh(grid[..., 0], grid[..., 2], score, cmap=cmap, vmin=0, vmax=1)
    ax.set_aspect('equal')

    # Plot true objects
    for i, obj in enumerate(objects):
        color = choose_color2(obj.classname)

        # Get corners of 3D bounding box
        corners = bbox_corners(obj)[:, [0, 2]]
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            ax.add_line(Line2D(*zip(start, end), c=color))
     
    ax.set_aspect('equal')
    # Format axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    # Return the modified axis object
    return ax

#For Display
def compute_iou(obj1, obj2):
    """Compute IoU between two bounding boxes.

    Args:
    obj1 (ndarray): a numpy array of shape (4,) representing the coordinates (x1, y1, x2, y2) of the first bounding box
    obj2 (ndarray): a numpy array of shape (4,) representing the coordinates (x1, y1, x2, y2) of the second bounding box

    Returns:
    iou (float): the Intersection over Union (IoU) between the two bounding boxes
    """
    corners1 = bbox_corners(obj1)[:4, [0, 2]]
    corners2 = bbox_corners(obj2)[:4, [0, 2]]

    x1 = min(corners1[0][0], corners1[1][0], corners1[2][0], corners1[3][0])
    y1 = min(corners1[0][1], corners1[1][1], corners1[2][1], corners1[3][1])
    x2 = max(corners1[0][0], corners1[1][0], corners1[2][0], corners1[3][0])
    y2 = max(corners1[0][1], corners1[1][1], corners1[2][1], corners1[3][1])
    
    x3 = min(corners2[0][0], corners2[1][0], corners2[2][0], corners2[3][0])
    y3 = min(corners2[0][1], corners2[1][1], corners2[2][1], corners2[3][1])
    x4 = max(corners2[0][0], corners2[1][0], corners2[2][0], corners2[3][0])
    y4 = max(corners2[0][1], corners2[1][1], corners2[2][1], corners2[3][1])
    
    intersection_w = max(0, min(x2, x4) - max(x1, x3))
    intersection_h = max(0, min(y2, y4) - max(y1, y3))
    intersection_area = intersection_w * intersection_h

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou

def nms(objects, iou_threshold=0.1):
    """Performs non-maximum suppression on a list of object detections.

    Args:
        objects (list): list of object detections, each containing a 'score' attribute
        iou_threshold (float): threshold for the IOU overlap between two detections to consider them redundant

    Returns:
        nms_objects (list): list of object detections after NMS, sorted in decreasing order of 'score'
    """
    # Sort by population
    sorted_objects = sorted(objects, key=lambda obj: obj.score, reverse=True)

    # Initialize list of detections after NMS
    nms_objects = []

    # Loop over detections
    while len(sorted_objects) > 0:
        # Keep detection with highest obj.score
        best_object = sorted_objects.pop(0)
        nms_objects.append(best_object)

        # Compute IOU between best_detection and remaining detections
        ious = [compute_iou(best_object, d) for d in sorted_objects]
        # Remove detections that overlap with best_detection
        sorted_objects = [d for i, d in enumerate(sorted_objects) if ious[i] <= iou_threshold]
    return nms_objects
