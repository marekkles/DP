import torch

__all__ = ['quality_score']

def quality_score(
    model: torch.nn.Module, 
    image: torch.Tensor, 
    device,
    T : int = 100,
    alpha : float = 130.0,
    r : float = 0.88,
) -> float:
    """
    Calculates the SER-FIQ score for a given aligned image using T passes.
    
    Parameters
    ----------
    model: torch.Module, Pytorch model of the network for serfiq
        Model has to have dropout enabled
    image: torch.Tensor, shape (c, h, w)
        Image, in RGB format.
    T : int, optional
        Amount of forward passes to use. The default is 100.
    alpha : float, optional
        Stretching factor, can be choosen to scale the score values
    r : float, optional
        Score displacement
    Returns
    -------
    SER-FIQ score : float
    """
    with torch.no_grad():
        repeated_image = image[None, :, :, :].to(device)
        embeddings = model(repeated_image,repeat_before_dropout=T)
        norm = torch.nn.functional.normalize(embeddings,dim=1)
        eucl_dist = torch.cdist(norm, norm)
        idx = torch.triu_indices(T,T,1)
        eucl_dist_triu = eucl_dist[idx[0],idx[1]]
        eucl_dist_mean = torch.mean(eucl_dist_triu)
        score = 2*(1 / ( 1 + eucl_dist_mean.exp() ))
        score = 1 / (1 + torch.exp( - (alpha * (score - r))))
    return score.item()

