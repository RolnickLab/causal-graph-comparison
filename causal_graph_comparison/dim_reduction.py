import numpy as np
from math import sqrt

def get_quadrant_centres(outputs, num_modes):
    """Find the centre coordinates of each quadrant in a 2D grid.
    
    Args:
        outputs (numpy.ndarray): The outputs array to determine dimensions
        num_modes (int): Number of modes (subdivisions) per side
        
    Returns:
        list: List of (x,y) coordinate tuples for the centre of each quadrant
    """
    # Calculate number of modes per side
    modes_per_side = int(sqrt(num_modes))
    
    # Calculate dimensions of the 2D array
    dimensions = int(sqrt(outputs.shape[-1]))
    
    # Calculate size of each quadrant
    quadrant_size = dimensions // modes_per_side
    
    # Calculate offset to get to centre of quadrant
    offset = quadrant_size // 2
    
    centres = []
    # Generate coordinates for centre of each quadrant
    for i in range(modes_per_side):
        for j in range(modes_per_side):
            x = i * quadrant_size + offset
            y = j * quadrant_size + offset
            centres.append((x, y))

    linear_indices = []
    for i, (x, y) in enumerate(centres):
        print("x, y: ", x, y)
        linear_idx = x * dimensions + y
        print("linear_idx: ", linear_idx)
        linear_indices.append(linear_idx)
            
    return centres, linear_indices

def spatial_subsample(outputs, num_modes):
    """Subsample the outputs to the given centres.

    Args:
        outputs (numpy.ndarray): The outputs to subsample, shape (num_samples, features)
        num_modes (int): Number of modes (subdivisions) per side
        
    Returns:
        numpy.ndarray: Subsampled outputs with shape (num_samples, modes_per_side, modes_per_side)
    """

    if len(outputs.shape) != 3:
        raise ValueError("Expecting 3D array of size (num_samples, num_timesteps, num_features), got shape: ", outputs.shape)
    
    # Get the centres for subsampling
    _, linear_indices = get_quadrant_centres(outputs, num_modes)

    subsampled = outputs[:,:, linear_indices]

    # print(subsampled.shape)

    return subsampled
 
if __name__ == "__main__":
    outputs = np.zeros((975, 20, 3600))

    ndim = int(sqrt(outputs.shape[-1]))
    centres, linear_indices = get_quadrant_centres(outputs, 4)
    outputs[:,:, linear_indices] = 1

    subsampled = spatial_subsample(outputs, 4)
    print(subsampled.shape)
    print(subsampled[0,2,:])