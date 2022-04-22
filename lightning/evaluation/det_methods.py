from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
from sklearn.metrics import det_curve

__all__ = [
    'det', 'det_for_irrs', 'fnmr_at_irr', 
    'generate_labels_scores', 'generate_labels_scores_quality',
    'generate_sorted_labels_scores_quality', 'eer_at_irr'
]

def det(
    label: np.array,
    score: np.array,
    count: int=100
) -> Tuple[np.array, np.array, np.array]:
    """Get Detection Tradeoff curve

    Args:
        label (np.array): Array of labels of 0 or 1 
        score (np.array): Scores for given label
        count (int, optional): Count of point on the curve. Defaults to 100.

    Returns:
        Tuple[np.array, np.array, np.array]: Tupple containing FMR, FNMR and Treashold
    """
    g = np.sum(label)
    i = np.sum(1 - label)
    index_list = np.argsort(score)
    label_sorted = label[index_list]
    score_sorted = score[index_list]
    graph = np.zeros((3,count))
    for k, j in enumerate(range(0, len(score), int(len(score)/count))):
        i_count = np.sum(1 - label_sorted[j:])
        g_count = np.sum(label_sorted[:j])
        fmr = i_count / i
        fnmr = g_count / g        
        graph[:,k] = (fmr, fnmr, score_sorted[j])
    return graph[0,:], graph[1,:], graph[2,:]

def det_for_irrs(
    sorted_labels: np.array, 
    sorted_scores: np.array, 
    irrs: Iterable[int]
) -> Tuple[float, Tuple[np.array, np.array, np.array]]:
    """Get det curve for specified input rejection rate

    Args:
        sorted_labels (np.array): Input labels sorted by their quality
        sorted_scores (np.array): Input score values sorted by their quality
        irrs (np.array): List of Input Rejection Rates
    Yields:
        Iterator[Tuple[float, Tuple[np.array, np.array, np.array]]]:
        Iterator for specific value of IRR and corresponding DET 
    """
    assert len(sorted_labels) == len(sorted_scores)
    for irr in irrs:
        idx = int((len(sorted_labels)-1)*irr)
        yield irr, det_curve(sorted_labels[idx:], sorted_scores[idx:])

def fnmr_at_irr(
    sorted_labels: np.array, sorted_scores: np.array,
    max_reject_rate: float = 0.15,
    number_of_samples: float  = 100,
    fmr_anchor: float  = 0.01
) -> Tuple[np.array, np.array]:
    """Get FNMR@IRR curve

    Args:
        sorted_labels (np.array): Labels of data sorted by their quality.
        sorted_scores (np.array): Distance scores sorted by their quality.
        max_reject_rate (float, optional): Maximum value of IRR. Defaults to 0.15.
        number_of_samples (float, optional): Number of samples to construct curve with. Defaults to 100.
        fmr_anchor (float, optional): Anchor point for FMR. Defaults to 0.01.

    Returns:
        Tuple[np.array, np.array]: Tupple with IRR and FNMR
    """
    assert len(sorted_labels) == len(sorted_scores), "Different input lengths"
    graph = []
    rejection_rates = np.arange(0, max_reject_rate, max_reject_rate/number_of_samples)[:-1]
    for irr, (fmr, fnmr, treashold) in det_for_irrs(sorted_labels, sorted_scores, rejection_rates):
        anchor_index = np.argmin(np.abs(np.array(fmr) - fmr_anchor))
        graph.append((irr, fnmr[anchor_index]))
    return tuple(zip(*graph))

def eer_at_irr(
    sorted_labels: np.array, sorted_scores: np.array,
    max_reject_rate: float = 0.15,
    number_of_samples: float  = 100
) -> Tuple[np.array, np.array]:

    assert len(sorted_labels) == len(sorted_scores), "Different input lengths"
    graph = []
    rejection_rates = np.arange(0, max_reject_rate, max_reject_rate/number_of_samples)[:-1]
    for irr, (fmr, fnmr, treashold) in det_for_irrs(sorted_labels, sorted_scores, rejection_rates):
        anchor_index = np.argmin(np.abs(np.array(fmr) - np.array(fnmr)))
        graph.append((irr, fnmr[anchor_index]))
    return tuple(zip(*graph))

def generate_labels_scores(
    pairs: Dict[Tuple[Any, Any], float],
    impostors: Dict[Tuple[Any, Any], float]
) -> Tuple[np.array, np.array, List[Tuple[Any, Any]]]:
    """
    Get numpy arrays for labels and scores form given pairs and impostors scores dictionaries.

    Args:
        pairs (Dict[Tuple[Any, Any], float]): Dictionary with pairs scores
        impostors (Dict[Tuple[Any, Any], float]): Dictionary with impostor scores

    Returns:
        Tuple[np.array, np.array, List[Tuple[Any, Any]]]: 
        Labels, Scores and pairs representing given indicies in returned numpy arrays
    """
    list_of_pairs = list(pairs.keys()) + list(impostors.keys())
    labels = np.zeros(len(list_of_pairs))
    labels[:len(pairs)] = 1
    scores = list(pairs.values()) + list(impostors.values())
    return np.array(labels), np.array(scores), list_of_pairs

def generate_labels_scores_quality(
    pairs: Dict[Tuple[Any, Any], float],
    impostors: Dict[Tuple[Any, Any], float],
    quality: Dict[Any, float],
    quality_reduction:str = "min"
) -> Tuple[np.array, np.array, np.array]:
    """Get numpy arrays for labels and scores form given pairs and impostors scores dictionaries.
    Arrays will be sorted by their quality.

    Args:
        pairs (Dict[Tuple[Any, Any], float]): Dictionary with pairs scores
        impostors (Dict[Tuple[Any, Any], float]): Dictionary with impostor scores
        quality (Dict[Any, float]): Dictionary for quality labels
        quality_reduction (str, optional): 
        Reduction method to use in order to get quality of a pair. Defaults to "min".
        Can be "min" or  "max"

    Raises:
        ValueError: If given quality reduction method is not supported

    Returns:
        Tuple[np.array, np.array, np.array, List[Tuple[Any, Any]]]: 
        Labels, Scores, Qualities, all sorted by their quality score
    """

    labels, scores, list_of_pairs = generate_labels_scores(pairs, impostors)
    
    a = np.array([quality[x[0]] for x in list_of_pairs])
    b = np.array([quality[x[1]] for x in list_of_pairs])

    if quality_reduction == "min":
        quality_scores = np.minimum(a, b)
    elif quality_reduction == "max":
        quality_scores = np.minimum(a, b)
    else:
        raise ValueError("Unknown quality redction method: %s" %quality_reduction)

    return labels, scores, quality_scores
    
def generate_sorted_labels_scores_quality(
    pairs: Dict[Tuple[Any, Any], float],
    impostors: Dict[Tuple[Any, Any], float],
    quality: Dict[Any, float],
    quality_reduction:str = "min"
) -> Tuple[np.array, np.array, np.array]:
    """Get numpy arrays for labels and scores form given pairs and impostors scores dictionaries.
    Arrays will be sorted by their quality.

    Args:
        pairs (Dict[Tuple[Any, Any], float]): Dictionary with pairs scores
        impostors (Dict[Tuple[Any, Any], float]): Dictionary with impostor scores
        quality (Dict[Any, float]): Dictionary for quality labels
        quality_reduction (str, optional): 
        Reduction method to use in order to get quality of a pair. Defaults to "min".
        Can be "min" or  "max"

    Raises:
        ValueError: If given quality reduction method is not supported

    Returns:
        Tuple[np.array, np.array, np.array, List[Tuple[Any, Any]]]: 
        Labels, Scores, Qualities, all sorted by their quality score
    """

    labels, scores, quality_scores = generate_labels_scores_quality(
        pairs, impostors, quality, quality_reduction=quality_reduction
    )
    idx = np.argsort(quality_scores)
    if quality_reduction == 'max':
        idx = np.flip(idx)
    return labels[idx], scores[idx], quality_scores[idx]
    