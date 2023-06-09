import math
import random
from PIL import ImageOps
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from utils import perspective

ObjectData = namedtuple('ObjectData', 
    ['classname','truncated', 'occlusion', 'position', 'dimensions', 'angle', 'score'])

def random_crop(image, calib, objects, output_size):
    """Random crop images.

    Args:
        image: PIL image
        calib (torch.tensor): calibration matrix
        objects (list): list of labels
        output_size (tuple): expected output size
    Returns:
        image: output PIL image
        calib (torch.tensor): calibration matrix
        cropped_objects (list): a list of labels after transformation
    """
    # Randomize bounding box coordinates
    width, height = image.size
    w_out, h_out = output_size
    left = random.randint(0, max(width - w_out, 0))
    upper = random.randint(0, max(height - h_out, 0))

    # Crop image
    right = left + w_out
    lower = upper + h_out
    image = image.crop((left, upper, right, lower))

    # Modify calibration matrix
    calib[0, 2] = calib[0, 2] - left                # cx' = cx - du
    calib[1, 2] = calib[1, 2] - upper               # cy' = cy - dv
    calib[0, 3] = calib[0, 3] - left * calib[2, 3]  # tx' = tx - du * tz
    calib[1, 3] = calib[1, 3] - upper * calib[2, 3] # ty' = ty - dv * tz

    # Include only visible objects
    cropped_objects = list()
    for obj in objects:
        upos = perspective(calib, calib.new(obj.position))[0]
        if upos >= 0 and upos < w_out:
            cropped_objects.append(obj)

    return image, calib, cropped_objects


def random_scale(image, calib, scale_range=(0.8, 1.2)):
    """Randomly scales the input image and corresponding calibration matrix.

    Args:
        image: PIL image
        calib (torch.tensor): calibration matrix
        objects (list): list of labels
    Returns:
        image: Scaled PIL image
        calib (torch.tensor): Scaled calibration matrix
    """

    scale = random.uniform(*scale_range)

    # Scale image
    width, height = image.size
    image = image.resize((int(width * scale), int(height * scale)))

    # Scale first two rows of calibration matrix
    calib[:2, :] *= scale

    return image, calib


def random_flip(image, calib, objects):
    """Randomly flips the input image and modifies the calibration matrix and object positions accordingly.

    Args:
        image: PIL image
        calib (torch.tensor): calibration matrix
        objects (list): list of labels
    Returns:
        image (PIL Image): The flipped image.
        calib (torch.Tensor): The modified calibration matrix.
        flipped_objects (list): A list of ObjectData instances with flipped x-positions.
    """
    if random.random() < 0.5:
        return image, calib, objects
    
    # Flip image
    image = ImageOps.mirror(image)

    # Modify calibration matrix
    width, _ = image.size
    calib[0, 2] = width - calib[0, 2]               # cx' = w - cx
    calib[0, 3] = width * calib[2, 3] - calib[0, 3] # tx' = w*tz - tx

    # Flip object x-positions
    flipped_objects = list()
    for obj in objects:
        position = [-obj.position[0]] + obj.position[1:]
        angle = math.atan2(math.sin(obj.angle), -math.cos(obj.angle))
        flipped_objects.append(ObjectData(
            obj.classname, position, obj.dimensions, angle, obj.score
        ))

    return image, calib, flipped_objects


def random_crop_grid(grid, objects, crop_size):
    """Randomly crops a grid of 3D points, making sure to include at least one object if possible.

    Args:
        grid (torch.Tensor): A grid of 3D points. Dimensions are (depth, width, 3).
        objects (list]): list of labels
        crop_size (Tuple[int, int]): The desired crop size, as a (width, depth) tuple.
    Returns:
        cropped_grid (torch.Tensor): A cropped grid of 3D points. Dimensions are (depth, width, 3).
    """

    # Get input and output dimensions
    grid_d, grid_w, _ = grid.size()
    crop_w, crop_d = crop_size

    # Try and find a crop that includes at least one object
    for _ in range(10): # Timeout after 10 attempts

        # Randomize offsets
        xoff = random.randrange(grid_w - crop_w) if crop_w < grid_w else 0
        zoff = random.randrange(grid_d - crop_d) if crop_d < grid_d else 0

        # Crop grid
        cropped_grid = grid[zoff:zoff+crop_d, xoff:xoff+crop_w].contiguous()

        # If there are no objects present, any random crop will do
        if len(objects) == 0:
            return cropped_grid
        
        # Get bounds
        minx, _, minz = cropped_grid.view(-1, 3).min(dim=0)[0]
        maxx, _, maxz = cropped_grid.view(-1, 3).max(dim=0)[0]

        # Search objects to see if any lie within the grid
        for obj in objects:
            objx, _, objz = obj.position

            # If any object overlaps with the grid, return
            if minx < objx < maxx and minz < objz < maxz:
                return cropped_grid

    return cropped_grid

    
def random_jitter_grid(grid, std):
    grid += torch.randn(3) * torch.tensor(std)
    return grid


class AugmentedObjectDataset(Dataset):

    def __init__(
        self, 
        dataset,
        image_size=(1080, 360),
        grid_size=(160, 160), 
        scale_range=(0.8, 1.2)
    ):
        self.dataset = dataset
        self.image_size = image_size
        self.grid_size = grid_size
        self.scale_range = scale_range
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        idx, image, calib, objects, grid = self.dataset[index]

        # Apply image augmentation
        image, calib = random_scale(image, calib, self.scale_range)
        image, calib, objects = random_crop(image, calib, objects, self.image_size)
        image, calib, objects = random_flip(image, calib, objects)

        # Augment grid
        grid = random_crop_grid(grid, objects, self.grid_size)

        return idx, image, calib, objects, grid