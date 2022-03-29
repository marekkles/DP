import numpy as np
import scipy.spatial as sp

__all__ = ["compute_distances", "pairs_impostor_scores"]

def compute_distances(doubles: list, vectors: dict, metric: str):
    distances = {}
    for a,b in doubles:
        a_vec = vectors[a]
        b_vec = vectors[b]
        d = sp.distance.__dict__[metric](a_vec, b_vec)
        distances[(a,b)] = d
    return distances

def pairs_impostor_scores(pairs: list, impostors: list, vectors: dict, metric: str):
    pairs_distances = compute_distances(pairs, vectors, metric)
    impostors_distances = compute_distances(impostors, vectors, metric)
    return {
        "metric": metric,
        "pairs": pairs_distances,
        "impostors": impostors_distances
    }
