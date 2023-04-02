import torch
from data.kitti import KittiObjectDataset
from data.augmentation import AugmentedObjectDataset
from visualization.visualization import draw_3d_boxes, vis_score
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Dataset unit-test
    # kitti_root = "/home/giang/Documents/KITTI"
    kitti_root = "/home/son/Documents/Project/KITTI"
    dataset = KittiObjectDataset(kitti_root)
    idx, image, calib, objects, grid = dataset[0]
    # Try some augmentation
    train_image_size = (1080, 360)
    train_grid_size = (120, 120)
    aug_dataset = AugmentedObjectDataset(
        dataset,
        train_image_size,
        train_grid_size
    )
    idx, image, calib, objects, grid = aug_dataset[245]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 8))
    #Visualize bounding box
    img = draw_3d_boxes(np.array(image), objects, calib)
    ax1.imshow(img)
    ax1.set_title('3D bounding box')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./results/3d_boxes.jpg", img_rgb)
    #Visualize score
    vis_score(image, calib, objects, grid, ax=ax2)
    ax2.set_title('Bird_eye_view')
    plt.savefig('./results/3d_boxes_with_bird_eye_view.png')
    plt.show()