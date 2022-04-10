import torch
import numpy as np
import pickle
import os

__all__ = [
    'name_gen', 'unpack_results', 
    'save_labeled_results', 'save_unlabeled_results'
]

def name_gen(timestamp: int):
    available_chars=[
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', 
        '5', '6', '7', '8', '9', '0'
    ]
    ts = timestamp
    name = []
    while ts != 0:
        idx = ts % len(available_chars)
        ts = ts // len(available_chars)
        name.append(available_chars[idx])
    return ''.join(name)

def unpack_results(data):
    results = {}
    #Unpack result form list of results into result containing lists
    for res in data:
        for k in res:
            if not k in results:
                results[k] = []
            #Convert data as neccessary
            if type(res[k]) == torch.Tensor:
                results[k].extend(
                    [i.cpu().detach().numpy() for i in res[k]]
                )
            elif type(res[k]) == tuple:
                results[k].extend(list(res[k]))
            elif type(res[k]) == list:
                results[k].extend(res[k])
    return results

def save_labeled_results(results, root_dir, run_name, dataset_name):
    results["label"] = [
        l.item() if type(l) is np.ndarray else l
        for l in results["label"]
    ]
    for field_name in results:
        if field_name == "label":
            continue

        data = dict(zip(results["label"], results[field_name]))
        pickle_file_path = os.path.join(
            root_dir, f"{field_name}-{run_name}-{dataset_name}.pickle"
        )
        with open(pickle_file_path, "wb") as f:
            pickle.dump(data, f)

def save_unlabeled_results(results, root_dir, run_name, dataset_name):
    for field_name in results:
        data = results[field_name]
        pickle_file_path = os.path.join(
            root_dir, f"{field_name}-{run_name}-{dataset_name}.pickle"
        )
        with open(pickle_file_path, "wb") as f:
            pickle.dump(data, f)
