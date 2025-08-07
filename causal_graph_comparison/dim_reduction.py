import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def get_quadrant_centres(outputs:np.ndarray, num_modes:int) -> tuple[list[tuple[int, int]], list[int]]:
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

def spatial_subsample(outputs:np.ndarray, num_modes:int) -> np.ndarray:
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

def normal_dist(x:np.ndarray, mean:float, sd:float) -> np.ndarray:
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def normalize_1d(data:np.ndarray) -> np.ndarray:
    return (data - data.min()) / (data.max() - data.min())

def create_gaussian_kernel(size:int, mean:float, sd:float) -> np.ndarray:
    x = np.linspace(-3, 3, size)
    normal_kernel = normal_dist(x, mean, sd)
    normal_kernel = normalize_1d(normal_kernel)
    return normal_kernel

def get_mean_modes(outputs:np.ndarray, num_modes:int, gaussian:bool=True) -> np.ndarray:
    """Get the mean of each quadrant to reduce dimensionality using a gaussian kernel.

    Args:
        outputs (numpy.ndarray): The outputs to subsample, shape (num_samples, num_timesteps, num_features)
        num_modes (int): Number of modes (subdivisions) per side
        
    Returns:
        numpy.ndarray: Mean of each quadrant with shape (num_samples, num_timesteps, num_modes)
    """
    if len(outputs.shape) != 3:
        raise ValueError("Expecting 3D array of size (num_samples, num_timesteps, num_features), got shape: ", outputs.shape)

    quadrant_size = outputs.shape[-1] // num_modes

    # divide square into num_modes quadrants
    quadrants = np.array_split(outputs, num_modes, axis=2)

    normal_kernel = create_gaussian_kernel(quadrant_size, mean = 0, sd = 1)
    
    if gaussian:
        # Take mean of each quadrant along the feature dimension (axis=2)
        mean_modes = np.array([np.average(quadrant, axis=2, weights=normal_kernel) for quadrant in quadrants])
    else:
        mean_modes = np.array([np.mean(quadrant, axis=2) for quadrant in quadrants])
    
    # Transpose to get shape (num_samples, num_timesteps, num_modes)
    mean_modes = np.transpose(mean_modes, (1, 2, 0))

    return mean_modes
 
if __name__ == "__main__":

    # ========Test spatial subsampling
    # outputs = np.zeros((975, 20, 3600))

    # ndim = int(sqrt(outputs.shape[-1]))
    # centres, linear_indices = get_quadrant_centres(outputs, 4)
    # outputs[:,:, linear_indices] = 1

    # subsampled = spatial_subsample(outputs, 4)
    # print(subsampled.shape)
    # print(subsampled[0,2,:])

    # ======== Test mean dim reduction

    outputs = np.zeros((975, 20, 1600), dtype=int)
    outputs[:,:,:] = np.arange(1, 1601)
    print("outputs.shape: ", outputs.shape)
    print("outputs[0,1,:]: ", outputs[0,1,:])

    mean_modes = get_mean_modes(outputs, 4)
    print("mean_modes.shape: ", mean_modes.shape)
    print("mean_modes: ", mean_modes[0,1,:])
 