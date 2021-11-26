from dcflowsim.network import network_elements


def get_tor_node_ids_of_flow(flow_obj):
    """
    Returns tor node ids of flow's source and destination. If they are Hosts it gets the physical neighbor
    Args:
        flow_obj:

    Returns:

    """
    src_node = flow_obj.source
    dst_node = flow_obj.destination
    if isinstance(dst_node, network_elements.Host):
        dst_node = list(dst_node.get_neighbors().values())[0]
    if isinstance(src_node, network_elements.Host):
        src_node = list(src_node.get_neighbors().values())[0]
    return src_node.node_id, dst_node.node_id


def get_tor_node_of_flow(flow_obj):
    src_node = flow_obj.source
    dst_node = flow_obj.destination
    if isinstance(dst_node, network_elements.Host):
        dst_node = list(dst_node.get_neighbors().values())[0]
    if isinstance(src_node, network_elements.Host):
        src_node = list(src_node.get_neighbors().values())[0]
    return src_node, dst_node
