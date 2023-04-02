import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import namedtuple
from utils import make_grid

ObjectData = namedtuple('ObjectData', 
    ['classname','truncated', 'occlusion', 'position', 'dimensions', 'angle', 'score'])

KITTI_CLASS_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                     'Cyclist', 'Tram', 'Misc', 'DontCare']

class KittiObjectDataset(Dataset):
    """KITTI Object dataset.

    Args:
        kitti_root (str): root of the data. 
        split (str): train or test set.
        grid_size (tuple): size of grid in meter. Default (80., 80.)
        grid_res (float): grid solution. Default is 0.5m.
        y_offset (float): offset of y axis.
    """
    def __init__(self, kitti_root, 
                split='train', 
                grid_size=(80., 80.), 
                grid_res=0.5, 
                y_offset=1.74):
        kitti_split = 'testing' if split == 'test' else 'training'
        self.root = os.path.join(kitti_root, 'object', kitti_split)
        
        # Read split indices from file
        split_file = kitti_root + '/ImageSets/{}.txt'.format(split)
        self.indices = read_split(split_file)
        self.grid_size = grid_size
        self.y_offset = y_offset
        self.grid_res = grid_res
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        """Get item from dataset.

        Args:
            index (int): idx of an image in split file.
        Returns:
            idx (int): idx of an image in data
            image: PIL image
            calib (torch.tensor): calib matrix with the shape (3, 4)
            objects (list): list of labels
            grid (torch.tensor): 3D grid with fixed y.
        """
        # Load image
        idx = self.indices[index]
        img_file = os.path.join(self.root, 'image_2/{:06d}.png'.format(idx))
        image = Image.open(img_file)

        # Load calibration matrix
        calib_file = os.path.join(self.root, 'calib/{:06d}.txt'.format(idx))
        calib = read_kitti_calib(calib_file)

        # Load annotations
        label_file = os.path.join(self.root, 'label_2/{:06d}.txt'.format(idx))
        objects = read_kitti_objects(label_file)

        # Make grid
        grid = make_grid(self.grid_size, 
                        (-self.grid_size[0]/2., self.y_offset, 0.), 
                        self.grid_res)

        return idx, image, calib, objects, grid


def read_split(filename):
    """Read a list of indices.

    Args:
        filename (str): name of file.
    Returns:
        List of indices to a subset of the KITTI training or testing sets
    """
    with open(filename) as f:
        return [int(val) for val in f]

def read_kitti_calib(filename):
    """Read the camera calibration matrix P2 from a text file.
    
    Args:
        filename (str): name of file.
    Returns:
        calib (torch.tensor): Calib file with the shape of (3, 4).    
    """

    with open(filename) as f:
        for line in f:
            data = line.split(' ')
            if data[0] == 'P2:':
                calib = torch.tensor([float(x) for x in data[1:13]])
                return calib.view(3, 4)
    
    raise Exception(
        'Could not find entry for P2 in calib file {}'.format(filename))
    
def read_kitti_objects(filename):
    """Read Kitti object. One row corresponds to one object

    Args:
        filename (str): name of file.
    Returns:
        objects (list): list of objects with each of object class 
            ['classname','truncated', 'occlusion', 'position', 'dimensions', 'angle', 'score']
    """
    objects = list()
    with open(filename, 'r') as fp:
        
        # Each line represents a single object
        for line in fp:
            objdata = line.split(' ')
            if not (14 <= len(objdata) <= 15): 
                raise IOError('Invalid KITTI object file {}'.format(filename))
            # Parse object data
            objects.append(ObjectData(
                classname=objdata[0],
                truncated = float(objdata[1]),
                occlusion = float(objdata[2]),
                dimensions=[float(objdata[10]), float(objdata[8]), float(objdata[9])],
                position=[float(p) for p in objdata[11:14]],
                angle=float(objdata[14]),
                score=float(objdata[15]) if len(objdata) == 16 else 1.
        ))

    return objects