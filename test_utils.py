from utils import make_grid
# export PYTHONPATH=$HOME/giang/Documents/Project_3D_Object_Detection/:$PYTHONPATH

if __name__ == "__main__":

    # 1. Test make grid. The generated grid looks like
    
    # z_axis    
    # ^
    # |
    # |
    # |
    # |
    # |
    # |0
    # < - - - - - 0 - - - - - > x_axis

    grid_size = (4.0, 4.0)
    y_offset = 1.0
    grid_offset = (-grid_size[0]/2, y_offset, 0)
    grid_res = 0.5

    # Grid shape (8, 8, 3)
    grid = make_grid(grid_size, grid_offset, grid_res)

    print(grid[2, 0, :])

    # 2. 