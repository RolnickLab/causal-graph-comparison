import numpy as np

def get_timelag(node: str) -> int:
    """Extracts the time lag from a node with naming "NX, TX".

    Args:
        node: The node name in format "NX, TX".

    Returns:
        The time lag as an integer.
    """
    return int(node.split("T")[1])

def binarize_array(array: np.ndarray) -> np.ndarray:
    """Binarize an array.

    Args:
        array: The array to binarize.

    Returns:
        The binarized array.
    """
    return (array > 0).astype(np.int8)