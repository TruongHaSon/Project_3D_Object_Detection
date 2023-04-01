import cv2
import numpy as np
import torch.nn.functional as F
from matplotlib.lines import Line2D
import torch
from utils import corners3d, rotate, perspective, bbox_corners

import cv2
import torch.nn.functional as F
from matplotlib.lines import Line2D
import matplotlib.patches as patches
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
    image = cv2.addWeighted(image, 1, rect, 0, 0)

    for k in range(4):
        i, j = k, (k + 1) % 4
        cv2.line(image, corners3d[i], corners3d[j], color, thickness, lineType=cv2.LINE_AA)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, corners3d[i], corners3d[j], color, thickness, lineType=cv2.LINE_AA)
        i, j = k, k + 4
        cv2.line(image, corners3d[i], corners3d[j], color, thickness, lineType=cv2.LINE_AA)

    return image


def draw_3d_boxes(img, objects, calib, label = 'train'):
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
        if object.score is not None: score = object.score
        if isinstance(score, torch.Tensor): score = round(score.item(), 2)
        name = object.classname
        if label == 'train':
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        corners_3d = corners3d(object, calib)

        # Draw 3d bounding box
        img = draw_projected_box3d(img, corners_3d, color)

        # Find location for label
        points = [(int(corners_3d[i, 0]), int(corners_3d[i, 1])) for i in range(4)]
        min_x = min(point[0] for point in points)
        max_y = max(point[1] for point in points)
          
        # Show label
        if label == 'train':
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
        color = 'tab:blue'

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

def vis_score_test(image, calib, objects_predict, objects, grid, cmap='binary', ax=None):
    score = torch.randn(1, 1, grid.shape[0]-1, grid.shape[1]-1)  # adjust size to match X and Y dimensions of grid
    score = score[0, 0] * 0

    grid = grid.cpu().detach().numpy()

    # Create a new axis if one is not provided
    if ax is None:
        _, ax = plt.subplots()

    # Plot scores
    ax.clear()
    ax.pcolormesh(grid[..., 0], grid[..., 2], score, cmap=cmap, vmin=0, vmax=1)
    ax.set_aspect('equal')

    # Plot true objects
    for i, obj in enumerate(objects_predict):
        color = 'tab:blue'  # Màu xanh dương

        # Get corners of 3D bounding box
        corners = bbox_corners(obj)[:, [0, 2]]
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            ax.add_line(Line2D(*zip(start, end), c=color))

    for i, obj in enumerate(objects):
        color = 'tab:red'  # Màu đỏ

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

def test(image, calib, objects_predict, objects, grid):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 8))
    #Visualize bounding box
    img = draw_3d_boxes(image[0].permute(1, 2, 0).cpu().numpy().copy(), objects_predict[0], calib[0], label = 'train')
    img = draw_3d_boxes(img, objects[0], calib[0], label = 'test')
    ax1.imshow(img)
    ax1.set_title('3D bounding box')
    
    # Add legend patches
    true_box_patch = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='r', facecolor='r', label='Ground truth')
    predict_box_patch = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='b', facecolor='b', label='Prediction')
    ax1.legend(handles=[true_box_patch, predict_box_patch], loc='lower left', fontsize='large')
    
    #Visualize score
    vis_score_test(image[0], calib[0], objects_predict[0], objects[0], grid[0], ax=ax2)
    ax2.set_title('Bird-eye view')
    plt.savefig('/content/drive/MyDrive/3d_object_detection/3d_boxes_with_score.png')
    plt.show()