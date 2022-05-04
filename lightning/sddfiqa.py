import pytorch_lightning as pl
from dataset import *
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.stats import wasserstein_distance
from statistics import mean
import pickle
import matplotlib.pyplot as plt
import argparse
import yaml
import os

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="ArcFace model evaluation :get: embeddings", add_help=add_help)
    parser.add_argument("--run-root", type=str, help="Run root")
    parser.add_argument("--dataset-root", type=str, help="Dataset root")
    return parser

def main(args):
    ds = IrisVerificationDatasetPseudo(args['dataset_root'], 1, subsets=args['dataset_subsets'])
    image_ids= np.arange(len(ds.annotations), dtype=np.int32)
    image_classes = np.zeros(len(ds.annotations), dtype=np.int32)
    image_ids_string = np.array([a['__image_id'] for a in ds.annotations])

    for image_idx, annotation in enumerate(ds.annotations):
        image_classes[image_idx] = annotation['__class_number']
    
    permute_dict = {} #image id: (pairs: samples of pairs, impostros: samples of impostors)
    for image_idx in tqdm(image_ids, "Generating random pairs and impostors:"):
        image_cls = image_classes[image_idx]
        image_id_str = image_ids_string[image_idx]

        image_ids_pair = image_ids[np.logical_and(image_classes == image_cls, image_ids != image_idx)]
        image_ids_impostor = image_ids[image_classes != image_cls]

        permute_dict[image_id_str] = (
            tuple(image_ids_string[np.random.choice(image_ids_pair, m*K)].tolist()),
            tuple(image_ids_string[np.random.choice(image_ids_impostor, m*K)].tolist()),
        )
    
    rand_vec = {}
    run_name=args['run_name']
    run_root=args['run_root']
    with open(os.path.join(run_root, f"embedding-{run_name}-iris_verification_pseudo.pickle"), 'rb') as f:
        rand_vec = pickle.load(f)

    batch_size=32
    vec_dim=512

    vector_batch = np.zeros((len(rand_vec), vec_dim))
    idx_to_id = {}
    id_to_idx = {}

    for idx, img in enumerate(rand_vec):
        id_to_idx[img] = idx
        idx_to_id[idx] = img
        vector_batch[idx] = rand_vec[img]

    vector_len = np.linalg.norm(vector_batch, axis=1)
    one_over_vector_len = 1/vector_len
    distances_v = np.zeros((len(permute_dict), K*m*2, 1))

    for start_id in tqdm(range(0,len(permute_dict),batch_size), "Calculating cosine distances"):#len(permute_dict)

        root_vectors = vector_batch[start_id:start_id+batch_size, :, np.newaxis]
        root_one_over = one_over_vector_len[start_id:start_id+batch_size, np.newaxis, np.newaxis]
        batch_vectors = np.zeros((batch_size, K*m*2, vec_dim))
        batch_one_over = np.zeros((batch_size, K*m*2, 1))

        for i in range(start_id, min(start_id+batch_size, len(permute_dict))):
            pair, impostor = permute_dict[idx_to_id[i]]
            idxs = np.array(
                [id_to_idx[p] for p in pair ] + [id_to_idx[i] for i in impostor], 
                dtype=np.int32
            )
            batch_vectors[i-start_id, :, :] = vector_batch[idxs]
            batch_one_over[i-start_id, :, :] = one_over_vector_len[idxs, np.newaxis]
            
            
        distances_v[start_id:start_id+batch_size, :] = ((batch_vectors @ root_vectors) * batch_one_over * root_one_over)

    q = {}
    for idx in tqdm(range(len(permute_dict)), "Calculating wasserstein distances"):
        w_distance = 0
        image_id=idx_to_id[idx]
        for i in range(0, K*m, m):
            p = distances_v[idx][i:i+m][0]
            n = distances_v[idx][K*m+i:K*m+i+m][0]
            w_distance += wasserstein_distance(p, n)
        w_distance = w_distance/K
        q[image_id] = w_distance
    with open(os.path.join(run_root, f"pseudolabels-{run_name}.pickle"), "wb") as f:
        pickle.dump(q, f)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    print("Loading dataset")
    with open(os.path.join(args.run_root, 'args.yaml'), 'rb') as f:
        args_file=yaml.load(f)
    args_file['run_root']=args.run_root
    args_file['dataset_root']=args.dataset_root
    main(args_file)
    