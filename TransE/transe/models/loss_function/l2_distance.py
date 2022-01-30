import torch


def l2_distance(head_entity, relationship_label, tail_entity):
    """
    This function returns l2 distance between head+relation and tail
    
    Parameters
    ----------
    head_entity: torch.FloatTensor
        the embedding of head entity
    relationship_label: torch.FloatTensor
        the embedding of relation entity
    tail_entity: torch.FloatTensor
        the embedding of tail entity
    
    Returns
    -------
    dist: torch.FloatTensor
        the l2 distance between them
    """
    dist = torch.cdist(head_entity+relationship_label, tail_entity)
    return dist[0]