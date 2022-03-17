import numpy as np
import scipy.spatial as sp

__all__ = ["compute_distances", "pairs_impostor_scores"]

def compute_distances(doubles, vectors, distances = ["euclidean", "cityblock", "cosine"]):
    distances = dict([(distance_name, []) for distance_name in distances])
    for a,b in doubles:
        a_vec = vectors[a]
        b_vec = vectors[b]
        for distance_name in distances:
            d = sp.distance.__dict__[distance_name](a_vec, b_vec)
            distances[distance_name].append(d)
    return distances
def pairs_impostor_scores(pairs, impostors, vectors, distances = ["euclidean", "cityblock", "cosine"]):
    pairs_distances = compute_distances(pairs, vectors, distances)
    impostors_distances = compute_distances(impostors, vectors, distances)
    return {
        "pairs_data": pairs, 
        "impostors_data": impostors, 
        "pairs":pairs_distances, 
        "impostors": impostors_distances
    }