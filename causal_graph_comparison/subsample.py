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
            
    return centres

def subsample_outputs(outputs, num_modes):
    """Subsample the outputs to the given centres.

    Args:
        outputs (numpy.ndarray): The outputs to subsample, shape (num_samples, features)
        num_modes (int): Number of modes (subdivisions) per side
        
    Returns:
        numpy.ndarray: Subsampled outputs with shape (num_samples, modes_per_side, modes_per_side)
    """
    # Get dimensions of outputs
    num_samples = outputs.shape[0]
    modes_per_side = int(sqrt(num_modes))
    
    # Get the centres for subsampling
    centres = get_quadrant_centres(outputs, num_modes)
    
    # Calculate dimensions of the 2D array
    dimensions = int(sqrt(outputs.shape[-1]))
    
    # Initialize array to store subsampled outputs
    subsampled = np.zeros((num_samples, modes_per_side, modes_per_side))
    print("Subsampled shape: ", subsampled.shape)

    for i, output in enumerate(outputs):
        # Reshape 1D output to 2D for indexing
        output_2d = output.reshape(dimensions, dimensions)

        for j, (x, y) in enumerate(centres):
            # Calculate row and column indices for the subsampled array
            row = j // modes_per_side
            col = j % modes_per_side
            
            # Extract value at the centre coordinates
            subsampled[i, row, col] = output_2d[x, y]

    return subsampled