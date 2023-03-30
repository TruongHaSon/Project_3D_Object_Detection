import torch
from data.encoder import ObjectEncoder, non_maximum_suppression
from data.kitti import KittiObjectDataset
from argparse import ArgumentParser
from data.augmentation import AugmentedObjectDataset
from torch.utils.data import DataLoader
from utils import collate
def parse_args():
    parser = ArgumentParser()

    # Data options
    parser.add_argument('-f')
    parser.add_argument('--root', type=str, default='/home/son/Documents/Project/KITTI',
                        help='root directory of the KITTI dataset')
    parser.add_argument('--grid-size', type=float, nargs=2, default=(80., 80.),
                        help='width and depth of validation grid, in meters')
    parser.add_argument('--train-grid-size', type=int, nargs=2, 
                        default=(120, 120),
                        help='width and depth of training grid, in pixels')
    parser.add_argument('--grid-jitter', type=float, nargs=3, 
                        default=[.25, .5, .25],
                        help='magn. of random noise applied to grid coords')
    parser.add_argument('--train-image-size', type=int, nargs=2, 
                        default=(1080, 360),
                        help='size of random image crops during training')
    parser.add_argument('--yoffset', type=float, default=1.74,
                        help='vertical offset of the grid from the camera axis')
    parser.add_argument('--nms-thresh', type=float, default=0.2,
                        help='minimum score for a positive detection')
    
    # Model options
    parser.add_argument('--grid-height', type=float, default=4.,
                        help='size of grid cells, in meters')
    parser.add_argument('-r', '--grid-res', type=float, default=0.5,
                        help='size of grid cells, in meters')
    parser.add_argument('--frontend', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='name of frontend ResNet architecture')
    parser.add_argument('--topdown', type=int, default=8,
                        help='number of residual blocks in topdown network')
    
    # Optimization options
    parser.add_argument('-l', '--lr', type=float, default=1e-7,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr-decay', type=float, default=0.99,
                        help='factor to decay learning rate by every epoch')
    parser.add_argument('--loss-weights', type=float, nargs=4, 
                        default=[1., 1., 1., 1.],
                        help="loss weighting factors for score, position,"\
                            " dimension and angle loss respectively")

    # Training options
    parser.add_argument('-e', '--epochs', type=int, default=600,
                        help='number of epochs to train for')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='mini-batch size for training')
    
    # Experiment options
    parser.add_argument('--name', type=str, default='test',
                        help='name of experiment')
    parser.add_argument('-s', '--savedir', type=str, 
                        default='/content/drive/MyDrive/3d_object_detection/backup',
                        help='directory to save experiments to')
    parser.add_argument('-g', '--gpu', type=int, nargs='*', default=[0],
                        help='ids of gpus to train on. Leave empty to use cpu')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='number of worker threads to use for data loading')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='number of epochs between validation runs')
    parser.add_argument('--print-iter', type=int, default=10,
                        help='print loss summary every N iterations')
    parser.add_argument('--vis-iter', type=int, default=50,
                        help='display visualizations every N iterations')
    return parser.parse_args()

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse command line arguments
    args = parse_args()

    # Create datasets
    train_data = KittiObjectDataset(args.root, 'train', args.grid_size, args.grid_res, args.yoffset)
    val_data = KittiObjectDataset(args.root, 'val', args.grid_size, args.grid_res, args.yoffset)

    # Apply data augmentation
    train_data = AugmentedObjectDataset(train_data, args.train_image_size, args.train_grid_size)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, 
        num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, 
        num_workers=args.workers,collate_fn=collate)
        
    # Create encoder
    encoder = ObjectEncoder()

    for i, (_, image, calib, objects, grid) in enumerate(train_loader):
        if i==1: break
        image, calib, grid = image.to(device), calib.to(device), grid.to(device)
        gt_encoded = encoder.encode_batch(objects, grid)
        print(gt_encoded[0].shape)
        print(gt_encoded[1].shape)
        print(gt_encoded[2].shape)
        print(gt_encoded[3].shape)
        gt = encoder.decode_batch(gt_encoded[0], gt_encoded[1], gt_encoded[2], gt_encoded[3], grid)
        print(gt)