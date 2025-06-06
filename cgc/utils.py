def get_timelag(node):
    """
    Extracts the time lag from a node with naming "NX, TX.
    :param node:
    :return:
    """
    return int(node.split("T")[1])
