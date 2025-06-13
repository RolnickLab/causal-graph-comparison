def get_timelag(node: str) -> int:
    """Extracts the time lag from a node with naming "NX, TX".

    Args:
        node: The node name in format "NX, TX".

    Returns:
        The time lag as an integer.
    """
    return int(node.split("T")[1])
